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
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
from rlinf.workers.sft.openpi_action_eval import (
    ActionMetricAccumulator,
    make_fixed_eval_noise,
    select_uniform_chunk_indices,
)


def split_episode_indices(
    episode_indices: list[int], test_episode_count: int, seed: int
) -> tuple[list[int], list[int]]:
    """Split complete episodes into deterministic train and test subsets."""
    episode_indices = sorted(set(episode_indices))
    if not 0 < test_episode_count < len(episode_indices):
        raise ValueError(
            "test_episode_count must be greater than zero and smaller than the "
            f"number of episodes ({len(episode_indices)}), got {test_episode_count}."
        )

    rng = np.random.default_rng(seed)
    test_episodes = sorted(
        int(index)
        for index in rng.choice(
            episode_indices, size=test_episode_count, replace=False
        ).tolist()
    )
    test_episode_set = set(test_episodes)
    train_episodes = [
        index for index in episode_indices if index not in test_episode_set
    ]
    return train_episodes, test_episodes


def fix_lerobot_episode_data_index(dataset: Any, episodes: list[int]) -> None:
    """Map filtered LeRobot episode bounds back to their original episode IDs."""
    episode_mapping = {
        episode: filtered_index
        for filtered_index, episode in enumerate(sorted(episodes))
    }
    old_from = dataset.episode_data_index["from"]
    old_to = dataset.episode_data_index["to"]
    index_size = max(episodes) + 1
    new_from = torch.full((index_size,), -1, dtype=old_from.dtype)
    new_to = torch.full((index_size,), -1, dtype=old_to.dtype)
    for episode, filtered_index in episode_mapping.items():
        new_from[episode] = old_from[filtered_index]
        new_to[episode] = old_to[filtered_index]
    dataset.episode_data_index["from"] = new_from
    dataset.episode_data_index["to"] = new_to


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            == SupportedModel.OPENPI_PYTORCH
        ):
            # DROID uses the upstream OpenPI LeRobot transforms.  Reuse the
            # same trajectory-level split/evaluation loader as the legacy
            # OpenPI wrapper; the JAX-aligned model accepts its Observation
            # dataclass through a field-preserving adapter.
            episode_split_cfg = self.cfg.data.get("episode_split")
            if episode_split_cfg is not None:
                repo_id = resolve_lerobot_repo_id(data_paths)
                if repo_id is None:
                    raise ValueError(
                        "OpenPI PyTorch DROID SFT requires a local LeRobot "
                        "dataset path or repo id."
                    )
                from rlinf.models.embodiment.openpi.dataconfig import (
                    get_openpi_config,
                )

                config = get_openpi_config(
                    self.cfg.actor.model.openpi.config_name,
                    model_path=self.cfg.actor.model.model_path,
                    batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                    repo_id=repo_id,
                    data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
                )
                return self._build_split_openpi_dataloader(
                    config,
                    repo_id,
                    episode_split_cfg,
                    eval_dataset=eval_dataset,
                )

            from rlinf.data.datasets.openpi_pytorch import (
                build_openpi_pytorch_sft_dataloader,
            )

            return build_openpi_pytorch_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            repo_id = resolve_lerobot_repo_id(data_paths)
            if repo_id is None:
                raise ValueError(
                    "OpenPI SFT requires data.train_data_paths to be set to a local "
                    "dataset path or LeRobot repo id."
                )

            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
            )
            episode_split_cfg = self.cfg.data.get("episode_split")
            if episode_split_cfg is not None:
                return self._build_split_openpi_dataloader(
                    config,
                    repo_id,
                    episode_split_cfg,
                    eval_dataset=eval_dataset,
                )
            data_loader = openpi_data_loader.create_data_loader(
                config,
                framework="pytorch",
                shuffle=not eval_dataset,
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
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO
        ]:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _build_split_openpi_dataloader(
        self,
        config: Any,
        repo_id: str,
        episode_split_cfg: Any,
        *,
        eval_dataset: bool,
    ):
        """Build an OpenPI loader restricted to complete train or test episodes."""
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        import openpi.training.data_loader as openpi_data_loader
        import openpi.transforms as transforms

        from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root

        dataset_root = resolve_lerobot_dataset_root(repo_id)
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
            repo_id, root=dataset_root
        )
        all_episodes = sorted(int(index) for index in dataset_meta.episodes)
        train_episodes, test_episodes = split_episode_indices(
            all_episodes,
            int(episode_split_cfg.get("test_episode_count", 1)),
            int(episode_split_cfg.get("seed", self.cfg.actor.seed)),
        )
        selected_episodes = test_episodes if eval_dataset else train_episodes
        episode_lengths = {
            episode: int(dataset_meta.episodes[episode]["length"])
            for episode in test_episodes
        }
        eval_indices, eval_offsets = select_uniform_chunk_indices(
            episode_lengths,
            test_episodes,
            config.model.action_horizon,
            int(episode_split_cfg.get("eval_chunks_per_episode", 10)),
        )
        self._write_episode_split(
            repo_id,
            train_episodes,
            test_episodes,
            int(episode_split_cfg.get("seed", self.cfg.actor.seed)),
            eval_offsets,
        )

        data_config = config.data.create(config.assets_dirs, config.model)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            root=dataset_root,
            episodes=selected_episodes,
            delta_timestamps={
                key: [
                    offset / dataset_meta.fps
                    for offset in range(config.model.action_horizon)
                ]
                for key in data_config.action_sequence_keys
            },
        )
        fix_lerobot_episode_data_index(dataset, selected_episodes)
        if data_config.prompt_from_task:
            dataset = openpi_data_loader.TransformedDataset(
                dataset,
                [transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
            )
        dataset = openpi_data_loader.transform_dataset(dataset, data_config)
        if eval_dataset:
            dataset = torch.utils.data.Subset(dataset, eval_indices)
            self._eval_noise = make_fixed_eval_noise(
                int(episode_split_cfg.get("seed", self.cfg.actor.seed)),
                len(eval_indices),
                config.model.action_horizon,
                config.model.action_dim,
            )
            self._eval_chunk_count = len(eval_indices)

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=not eval_dataset,
            seed=int(config.seed),
            drop_last=not eval_dataset,
        )
        local_batch_size = (
            self.eval_batch_size if eval_dataset else self.micro_batch_size
        )
        if eval_dataset and sampler.num_samples % local_batch_size != 0:
            raise ValueError(
                f"Per-rank evaluation samples ({sampler.num_samples}) must be "
                f"divisible by eval_batch_size ({local_batch_size})."
            )
        torch_data_loader = openpi_data_loader.TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            sampler=sampler,
            num_batches=(
                sampler.num_samples // local_batch_size if eval_dataset else None
            ),
            num_workers=config.num_workers,
            seed=config.seed,
            framework="pytorch",
        )
        data_loader = openpi_data_loader.DataLoaderImpl(data_config, torch_data_loader)
        subset_name = "test" if eval_dataset else "train"
        logging.info(
            "OpenPI %s split: %d episodes, %d samples on rank %d",
            subset_name,
            len(selected_episodes),
            len(dataset),
            self._rank,
        )
        return data_loader, data_config

    def _write_episode_split(
        self,
        repo_id: str,
        train_episodes: list[int],
        test_episodes: list[int],
        seed: int,
        eval_offsets: dict[int, list[int]],
    ) -> None:
        if self._rank != 0:
            return
        split_path = (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
            / "episode_split.json"
        )
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(
            json.dumps(
                {
                    "dataset": repo_id,
                    "seed": seed,
                    "train_episodes": train_episodes,
                    "test_episodes": test_episodes,
                    "eval_chunk_offsets": {
                        str(episode): offsets
                        for episode, offsets in eval_offsets.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )

    def run_eval(self) -> dict[str, float]:
        """Evaluate without perturbing the RNG stream used by later training."""
        rng_state = get_rng_state()
        was_training = self.model.training
        try:
            return self._run_decoded_action_eval()
        finally:
            set_rng_state(rng_state)
            self.model.train(was_training)

    def _run_decoded_action_eval(self) -> dict[str, float]:
        """Evaluate deterministic decoded action chunks over held-out episodes."""
        assert self.eval_data_loader is not None, "eval_data_loader is not set"
        if not hasattr(self, "_eval_noise"):
            raise RuntimeError("Decoded action evaluation noise was not initialized.")

        import openpi.transforms as openpi_transforms

        control_frequency_hz = float(
            self.cfg.actor.model.openpi_data.control_frequency_hz
        )
        unnormalize = openpi_transforms.Unnormalize(
            self.eval_data_config.norm_stats,
            use_quantiles=self.eval_data_config.use_quantile_norm,
        )
        rank_noise = self._eval_noise[self._rank :: self._world_size]
        noise_offset = 0
        accumulator = ActionMetricAccumulator(control_frequency_hz)

        # The runner consumes worker timing by the public RPC name (``run_eval``).
        # This implementation delegates to a helper, so relying on
        # ``worker_timer``'s caller-name inference would record
        # ``_run_decoded_action_eval`` and make ``Handle.consume_duration`` fail.
        with self.worker_timer("run_eval"), torch.no_grad():
            self.model.eval()
            for batch in self.eval_data_loader:
                observation, expert_actions = batch
                batch_size = int(expert_actions.shape[0])
                noise = torch.from_numpy(
                    rank_noise[noise_offset : noise_offset + batch_size]
                )
                with self.amp_context:
                    predicted_actions = self.model(
                        forward_type=ForwardType.ACTION_SAMPLE,
                        data=batch,
                        noise=noise,
                    )

                predicted_normalized = predicted_actions.detach().float().cpu().numpy()
                expert_normalized = expert_actions.detach().float().cpu().numpy()
                normalized_state = observation.state.detach().float().cpu().numpy()
                predicted_native_data = unnormalize(
                    {
                        "state": normalized_state,
                        "actions": predicted_normalized,
                    }
                )
                expert_native_data = unnormalize(
                    {
                        "state": normalized_state,
                        "actions": expert_normalized,
                    }
                )
                accumulator.update(
                    predicted_normalized,
                    expert_normalized,
                    predicted_native_data["actions"],
                    expert_native_data["actions"],
                    predicted_native_data["state"],
                )
                noise_offset += batch_size

            if noise_offset != len(rank_noise):
                raise RuntimeError(
                    f"Rank {self._rank} evaluated {noise_offset} chunks, "
                    f"expected {len(rank_noise)}."
                )
            totals = torch.as_tensor(
                accumulator.as_totals(), dtype=torch.float64, device=self.device
            )
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
            metrics = ActionMetricAccumulator.metrics_from_totals(
                totals.detach().cpu().numpy()
            )
            if int(metrics["num_chunks"]) != self._eval_chunk_count:
                raise RuntimeError(
                    f"Evaluated {metrics['num_chunks']} global chunks, "
                    f"expected {self._eval_chunk_count}."
                )
            return metrics

    def get_eval_model_output(self, batch: dict[str, Any]):
        loss, _ = self.get_train_model_output(batch)
        return loss.detach()

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI_PYTORCH:
            if self.cfg.data.get("episode_split") is not None:
                num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
                return max(1, num_batches // self.gradient_accumulation)
            return max(1, len(self.data_loader) // self.gradient_accumulation)
        if model_type == SupportedModel.OPENPI:
            num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
            return max(1, num_batches // self.gradient_accumulation)
        return super().get_max_steps_per_epoch()

    @staticmethod
    def _openpi_pytorch_dataloader(openpi_dataloader: Any):
        """Unwrap OpenPI `DataLoaderImpl` to the inner PyTorch DataLoader.

        OpenPI torch path:
          DataLoaderImpl._data_loader -> TorchDataLoader
          TorchDataLoader._data_loader / .torch_loader -> torch.utils.data.DataLoader

        """
        torch_data_loader = getattr(openpi_dataloader, "_data_loader", None)
        pytorch_dl = getattr(torch_data_loader, "_data_loader", None) or getattr(
            torch_data_loader, "torch_loader", None
        )
        if pytorch_dl is None:
            raise TypeError(
                "OpenPI dataloader does not expose an inner torch DataLoader; cannot infer steps per epoch from len()."
            )
        return pytorch_dl
