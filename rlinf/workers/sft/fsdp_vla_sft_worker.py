# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils import _pytree

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            == SupportedModel.OPENPI_RLINF
        ):
            from rlinf.data.datasets.openpi_rlinf import (
                build_openpi_rlinf_sft_dataloader,
            )

            return build_openpi_rlinf_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: dict[str, Any]):
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            == SupportedModel.OPENPI_RLINF
        ):
            with self.amp_context:
                output = self.model(forward_type=ForwardType.SFT, data=batch)
            return output["loss"] if isinstance(output, dict) else output
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            batch_data = _pytree.tree_map(
                lambda x: (
                    torch.as_tensor(x, device=self.device).contiguous().clone()
                    if isinstance(x, torch.Tensor)
                    else x
                ),
                batch,
            )
            with self.amp_context:
                losses_dict = self.model(forward_type=ForwardType.SFT, data=batch_data)
            return losses_dict["loss"]
        observation, actions = batch

        register_pytree_dataclasses(observation)
        observation = _pytree.tree_map(
            lambda x: (
                torch.as_tensor(x, device=self.device).contiguous().clone()
                if x is not None
                else x
            ),
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        with self.amp_context:
            losses = self.model(
                forward_type=ForwardType.SFT,
                data={"observation": observation, "actions": actions},
            )

        # train model return the loss
        return losses

    def get_max_steps_per_epoch(self) -> int:
        """Count real dataset epochs for the new OpenPI SFT path only."""
        if self.data_loader is None:
            return 0
        from rlinf.data.datasets.openpi_rlinf import (
            get_official_openpi_sft_num_batches,
            is_official_openpi_sft_dataloader,
        )

        num_batches = (
            get_official_openpi_sft_num_batches(self.data_loader)
            if is_official_openpi_sft_dataloader(self.data_loader)
            else len(self.data_loader)
        )
        return max(1, num_batches // self.gradient_accumulation)

    def _has_stateful_openpi_loader(self) -> bool:
        return (
            self.cfg.actor.model.model_type == SupportedModel.OPENPI_RLINF.value
            and callable(getattr(self.data_loader, "state_dict", None))
            and callable(getattr(self.data_loader, "load_state_dict", None))
        )

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)
        if self._has_stateful_openpi_loader():
            for filename, state in (
                ("data.pt", self.data_loader.state_dict()),
                ("rng.pt", get_rng_state()),
            ):
                all_states = [None] * self._world_size
                torch.distributed.all_gather_object(all_states, state)
                if self._rank == 0:
                    torch.save(all_states, os.path.join(save_path, filename))
                torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)
        if self._has_stateful_openpi_loader():
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            self.data_loader.load_state_dict(all_states[self._rank])
            self.data_iter = iter(self.data_loader)
            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])
            torch.distributed.barrier()
