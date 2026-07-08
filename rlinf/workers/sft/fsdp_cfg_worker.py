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

"""FSDP CFG Worker for Classifier-Free Guidance training.

Extends FSDPSftWorker with pre-computed advantage labels and
CfgMixtureDataset for weighted sampling across datasets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils._pytree import tree_map

from rlinf.data.datasets.recap.cfg_model import (
    AdvantagePreservingDataset,
    CFGDataLoaderImpl,
    CfgMixtureDataset,
    TokenizePromptWithGuidance,
)
from rlinf.data.datasets.recap.utils import (
    cast_image_features,
)
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

# Suppress libdav1d/ffmpeg verbose logging
try:
    import av

    av.logging.set_level(av.logging.FATAL)
except ImportError:
    pass


class FSDPCfgWorker(FSDPSftWorker):
    """FSDP worker for CFG (Classifier-Free Guidance) training.

    Extends FSDPSftWorker to load datasets with pre-computed advantages,
    use CfgMixtureDataset for weighted sampling, and pass advantage
    labels to model.forward for guidance selection.
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        FSDPModelManager.__init__(self, cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.global_batch_size = self.cfg.actor.global_batch_size
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.eval_batch_size = self.cfg.actor.get("eval_batch_size", 1)

        assert (
            self.global_batch_size % (self.micro_batch_size * self._world_size) == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"
        self.gradient_accumulation = (
            self.global_batch_size // self.micro_batch_size // self._world_size
        )

        self.data_loader, self.eval_data_loaders, self.data_config = (
            self.build_dataloader()
        )
        self.data_iter = iter(self.data_loader)

        self.global_step = 0
        self._data_epoch = 0
        self._data_iter_offset = 0

    @staticmethod
    def _load_advantages_lookup(
        data_path: str,
        advantage_tag: str | None = None,
        required: bool = True,
    ) -> dict[tuple[int, int], bool]:
        """Load advantage lookup from meta/advantages_{tag}.parquet or meta/advantages.parquet.

        Args:
            data_path: Path to LeRobot dataset.
            advantage_tag: Advantage tag name. If None, loads meta/advantages.parquet.
            required: If False, return an empty lookup when the sidecar is missing.

        Returns:
            Dict mapping (episode_index, frame_index) -> bool.
        """
        import pandas as pd

        if advantage_tag:
            meta_path = Path(data_path) / "meta" / f"advantages_{advantage_tag}.parquet"
        else:
            meta_path = Path(data_path) / "meta" / "advantages.parquet"

        if not meta_path.exists():
            if not required:
                return {}
            raise FileNotFoundError(
                f"Advantage file not found: {meta_path}. "
                f"Run compute_advantages.py first."
            )

        adv_df = pd.read_parquet(meta_path)

        lookup = dict(
            zip(
                zip(
                    adv_df["episode_index"].values.astype(int).tolist(),
                    adv_df["frame_index"].values.astype(int).tolist(),
                ),
                adv_df["advantage"].values.astype(bool).tolist(),
            )
        )
        return lookup

    @staticmethod
    def _positive_advantages_lookup(base_dataset: Any) -> dict[tuple[int, int], bool]:
        """Build an all-positive advantage lookup for eval-only diagnostics."""
        hf_dataset = AdvantagePreservingDataset._get_hf_dataset(base_dataset)
        if hf_dataset is None:
            raise ValueError(
                "Cannot access HF dataset to build default positive advantages."
            )
        return {
            (int(ep), int(fr)): True
            for ep, fr in zip(hf_dataset["episode_index"], hf_dataset["frame_index"])
        }

    def build_dataloader(self):
        """Build CFG dataloader with advantage-weighted sampling across datasets."""
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        import openpi.training.data_loader as openpi_data_loader
        import openpi.transforms as transforms

        from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        data_cfg = self.cfg.get("data", {})
        openpi_cfg = self.cfg.actor.model.openpi
        advantage_tag = data_cfg.get("advantage_tag", None)

        datasets_config = data_cfg.get("train_data_paths", [])
        if not datasets_config:
            raise ValueError(
                "At least one dataset must be provided in data.train_data_paths. "
                "Each dataset should have 'dataset_path' and optionally 'episodes' and 'weight' fields."
            )

        first_path = datasets_config[0]["dataset_path"]
        config = get_openpi_config(
            openpi_cfg.config_name,
            model_path=self.cfg.actor.model.model_path,
            batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            repo_id=first_path,
            data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
        )
        data_config = config.data.create(config.assets_dirs, config.model)

        model_transforms = self._build_model_transforms(data_config)
        norm_stats = data_config.norm_stats or {}

        def _build_dataset(ds_config: Any, *, eval_dataset: bool = False):
            data_path = ds_config["dataset_path"]
            dataset_root = resolve_lerobot_dataset_root(data_path)
            episodes = ds_config.get("episodes")

            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
                data_path, root=dataset_root
            )
            base_dataset = lerobot_dataset.LeRobotDataset(
                data_path,
                root=dataset_root,
                episodes=episodes,
                delta_timestamps={
                    key: [
                        t / dataset_meta.fps for t in range(config.model.action_horizon)
                    ]
                    for key in data_config.action_sequence_keys
                },
            )

            base_dataset.hf_dataset = cast_image_features(base_dataset.hf_dataset)

            if episodes is not None:
                self._fix_episode_data_index(base_dataset, episodes)

            if data_config.prompt_from_task:
                base_dataset = openpi_data_loader.TransformedDataset(
                    base_dataset,
                    [transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
                )

            # RepackTransform strips all keys except OpenPI required ones,
            # so AdvantagePreservingDataset is needed to restore the advantage field.
            transforms_list = [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *model_transforms,
            ]
            transformed_dataset = openpi_data_loader.TransformedDataset(
                base_dataset, transforms_list
            )

            dataset_advantage_tag = ds_config.get("advantage_tag", advantage_tag)
            advantages_lookup = self._load_advantages_lookup(
                data_path,
                dataset_advantage_tag,
                required=not eval_dataset,
            )
            if eval_dataset and not advantages_lookup:
                advantages_lookup = self._positive_advantages_lookup(base_dataset)
                if self._rank == 0:
                    self.log_info(
                        f"No eval advantages sidecar found for {data_path}; "
                        "using all-positive guidance labels for eval."
                    )
            if self._rank == 0:
                adv_filename = (
                    f"advantages_{dataset_advantage_tag}.parquet"
                    if dataset_advantage_tag
                    else "advantages.parquet"
                )
                self.log_info(
                    f"Loaded advantages from "
                    f"meta/{adv_filename} ({len(advantages_lookup)} entries)"
                )

            final_dataset = AdvantagePreservingDataset(
                base_dataset=base_dataset,
                transformed_dataset=transformed_dataset,
                advantages_lookup=advantages_lookup,
            )
            return final_dataset

        datasets_with_weights = []
        for ds_config in datasets_config:
            weight = ds_config.get("weight", 1.0)
            final_dataset = _build_dataset(ds_config, eval_dataset=False)

            datasets_with_weights.append((final_dataset, weight))

            if self._rank == 0:
                self.log_info(
                    f"Loaded dataset: {ds_config['dataset_path']} "
                    f"({len(final_dataset)} samples, weight={weight})"
                )

        combined_dataset = CfgMixtureDataset(
            datasets=datasets_with_weights,
            mode="train",
            balance_dataset_weights=data_cfg.get("balance_dataset_weights", True),
            seed=data_cfg.get("seed", 42),
        )

        torch_data_loader = self._create_torch_dataloader(
            combined_dataset, config, openpi_data_loader
        )

        data_loader = CFGDataLoaderImpl(data_config, torch_data_loader)

        eval_data_loaders = []
        for eval_config in data_cfg.get("eval_data_paths", []) or []:
            eval_config = dict(eval_config)
            eval_path = eval_config.get("dataset_path")
            if not eval_path:
                continue
            eval_dataset = _build_dataset(eval_config, eval_dataset=True)
            max_samples = eval_config.get("max_samples")
            if max_samples is not None:
                eval_dataset = torch.utils.data.Subset(
                    eval_dataset,
                    range(min(int(max_samples), len(eval_dataset))),
                )
            eval_loader = self._create_torch_dataloader(
                eval_dataset,
                config,
                openpi_data_loader,
                shuffle=False,
                drop_last=False,
                num_workers=int(
                    eval_config.get(
                        "num_workers",
                        data_cfg.get(
                            "eval_num_workers",
                            data_cfg.get("num_workers", config.num_workers),
                        ),
                    )
                ),
            )
            eval_name = eval_config.get("name", Path(eval_path).name)
            eval_data_loaders.append(
                (eval_name, CFGDataLoaderImpl(data_config, eval_loader))
            )
            if self._rank == 0:
                self.log_info(
                    f"Loaded eval dataset: {eval_path} "
                    f"({len(eval_dataset)} samples, name={eval_name})"
                )

        return data_loader, eval_data_loaders, data_loader.data_config()

    def _build_model_transforms(self, data_config: Any) -> list:
        """Replace TokenizePrompt with TokenizePromptWithGuidance in model transforms."""
        tokenizer = None
        for t in data_config.model_transforms.inputs:
            if hasattr(t, "tokenizer"):
                tokenizer = t.tokenizer
                break

        if tokenizer is None:
            raise ValueError("Cannot find tokenizer in model_transforms")

        model_transforms = []
        for t in data_config.model_transforms.inputs:
            if type(t).__name__ == "TokenizePrompt":
                model_transforms.append(
                    TokenizePromptWithGuidance(
                        tokenizer=tokenizer,
                        discrete_state_input=getattr(t, "discrete_state_input", False),
                    )
                )
            else:
                model_transforms.append(t)

        return model_transforms

    def _fix_episode_data_index(self, dataset: Any, episodes: list) -> None:
        """Fix LeRobotDataset episode_data_index when using specific episodes.

        LeRobotDataset has a bug where episode_data_index doesn't match the
        original episode indices when filtering by episodes. This fixes that.
        """
        ep_idx_mapping = {ep: i for i, ep in enumerate(sorted(episodes))}
        max_ep_idx = max(episodes) + 1

        old_from = dataset.episode_data_index["from"]
        old_to = dataset.episode_data_index["to"]

        new_from = torch.full((max_ep_idx,), -1, dtype=old_from.dtype)
        new_to = torch.full((max_ep_idx,), -1, dtype=old_to.dtype)

        for orig_ep, new_idx in ep_idx_mapping.items():
            new_from[orig_ep] = old_from[new_idx]
            new_to[orig_ep] = old_to[new_idx]

        dataset.episode_data_index["from"] = new_from
        dataset.episode_data_index["to"] = new_to

    def _create_torch_dataloader(
        self,
        dataset: Any,
        config: Any,
        openpi_data_loader: Any,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int | None = None,
    ) -> Any:
        """Create PyTorch DataLoader with distributed sampler."""
        batch_size = config.batch_size
        sampler = None

        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // self._world_size
        else:
            local_batch_size = batch_size

        # Use data config overrides if available, otherwise fall back to OpenPI defaults.
        data_cfg = self.cfg.get("data", {})
        if num_workers is None:
            num_workers = int(data_cfg.get("num_workers", config.num_workers))
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    def run_training(self):
        """Run one training step with advantage-based CFG guidance."""
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()

            assert (
                self.cfg.actor.global_batch_size
                % (self.cfg.actor.micro_batch_size * self._world_size)
                == 0
            ), "global_batch_size is not divisible by micro_batch_size * world_size"

            self.gradient_accumulation = (
                self.cfg.actor.global_batch_size
                // self.cfg.actor.micro_batch_size
                // self._world_size
            )

            metrics = {}
            avg_loss = 0.0

            for idx in range(self.gradient_accumulation):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )

                try:
                    observation, actions, advantage = next(self.data_iter)
                except StopIteration:
                    self._data_epoch = getattr(self, "_data_epoch", 0) + 1
                    self._current_epoch = self._data_epoch
                    self._data_iter_offset = 0
                    self.data_loader.set_epoch(self._data_epoch)
                    self.data_iter = iter(self.data_loader)
                    observation, actions, advantage = next(self.data_iter)
                self._data_iter_offset += 1

                register_pytree_dataclasses(observation)
                observation = tree_map(
                    lambda x: (
                        torch.as_tensor(x)
                        .contiguous()
                        .to(self.device, non_blocking=True)
                    ),
                    observation,
                )
                actions = actions.to(torch.float32).to(self.device, non_blocking=True)
                advantage = advantage.to(self.device, non_blocking=True)

                with self.amp_context:
                    loss, metrics_data = self.model(
                        data={
                            "observation": observation,
                            "actions": actions,
                            "advantage": advantage,
                        },
                    )
                    loss = loss.mean()

                loss = loss / self.gradient_accumulation
                avg_loss += loss.detach().item()
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

                if metrics_data is not None:
                    append_to_dict(metrics, metrics_data)

            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            lr_value = (
                lr_list[0] if len(lr_list) > 0 else self.optimizer.param_groups[0]["lr"]
            )
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "loss": avg_loss,
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                },
            )

            self.lr_scheduler.step()

            count_keys = {
                "conditional_count",
                "unconditional_count",
                "positive_label_count",
                "negative_label_count",
                "positive_conditional_count",
                "positive_unconditional_count",
                "negative_conditional_count",
                "negative_unconditional_count",
            }
            loss_sum_keys = {
                "conditional_loss_sum",
                "unconditional_loss_sum",
                "positive_conditional_loss_sum",
                "positive_unconditional_loss_sum",
                "negative_conditional_loss_sum",
                "negative_unconditional_loss_sum",
            }
            special_keys = count_keys | loss_sum_keys
            has_cfg_metrics = any(k in metrics for k in special_keys)

            if has_cfg_metrics:
                sum_m = {k: np.sum(v) for k, v in metrics.items() if k in special_keys}
                mean_m = {
                    k: np.mean(v) for k, v in metrics.items() if k not in special_keys
                }
                sum_m = all_reduce_dict(sum_m, op=torch.distributed.ReduceOp.SUM)
                mean_m = all_reduce_dict(mean_m, op=torch.distributed.ReduceOp.AVG)

                total = sum_m.get("conditional_count", 0) + sum_m.get(
                    "unconditional_count", 0
                )
                if total > 0:
                    mean_m["conditional_ratio"] = (
                        sum_m.get("conditional_count", 0) / total
                    )
                    mean_m["unconditional_ratio"] = (
                        sum_m.get("unconditional_count", 0) / total
                    )
                    mean_m["positive_label_ratio"] = (
                        sum_m.get("positive_label_count", 0) / total
                    )
                    mean_m["negative_label_ratio"] = (
                        sum_m.get("negative_label_count", 0) / total
                    )
                    mean_m["positive_conditional_ratio"] = (
                        sum_m.get("positive_conditional_count", 0) / total
                    )
                    mean_m["positive_unconditional_ratio"] = (
                        sum_m.get("positive_unconditional_count", 0) / total
                    )
                    mean_m["negative_conditional_ratio"] = (
                        sum_m.get("negative_conditional_count", 0) / total
                    )
                    mean_m["negative_unconditional_ratio"] = (
                        sum_m.get("negative_unconditional_count", 0) / total
                    )

                positive_total = sum_m.get("positive_label_count", 0)
                if positive_total > 0:
                    mean_m["positive_effective_conditional_ratio"] = (
                        sum_m.get("positive_conditional_count", 0) / positive_total
                    )
                    mean_m["positive_effective_unconditional_ratio"] = (
                        sum_m.get("positive_unconditional_count", 0) / positive_total
                    )

                negative_total = sum_m.get("negative_label_count", 0)
                if negative_total > 0:
                    mean_m["negative_effective_conditional_ratio"] = (
                        sum_m.get("negative_conditional_count", 0) / negative_total
                    )
                    mean_m["negative_effective_unconditional_ratio"] = (
                        sum_m.get("negative_unconditional_count", 0) / negative_total
                    )

                loss_map = {
                    "conditional_loss": (
                        "conditional_loss_sum",
                        "conditional_count",
                    ),
                    "unconditional_loss": (
                        "unconditional_loss_sum",
                        "unconditional_count",
                    ),
                    "positive_conditional_loss": (
                        "positive_conditional_loss_sum",
                        "positive_conditional_count",
                    ),
                    "positive_unconditional_loss": (
                        "positive_unconditional_loss_sum",
                        "positive_unconditional_count",
                    ),
                    "negative_conditional_loss": (
                        "negative_conditional_loss_sum",
                        "negative_conditional_count",
                    ),
                    "negative_unconditional_loss": (
                        "negative_unconditional_loss_sum",
                        "negative_unconditional_count",
                    ),
                }
                for metric_name, (loss_key, count_key) in loss_map.items():
                    count = sum_m.get(count_key, 0)
                    if count > 0:
                        mean_m[metric_name] = sum_m.get(loss_key, 0) / count

                train_metrics = mean_m
            else:
                train_metrics = all_reduce_dict(
                    {k: np.mean(v) for k, v in metrics.items()},
                    op=torch.distributed.ReduceOp.AVG,
                )

            return train_metrics

    def _prepare_cfg_batch(self, observation, actions, advantage):
        """Move a CFG batch to the worker device."""
        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda x: (
                torch.as_tensor(x).contiguous().to(self.device, non_blocking=True)
            ),
            observation,
        )
        actions = actions.to(torch.float32).to(self.device, non_blocking=True)
        advantage = advantage.to(self.device, non_blocking=True)
        return observation, actions, advantage

    @staticmethod
    def _action_error_metrics(
        predicted_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> dict[str, float]:
        """Compute MSE/MAE between predicted and target action chunks."""
        horizon = min(predicted_actions.shape[1], target_actions.shape[1])
        action_dim = min(predicted_actions.shape[2], target_actions.shape[2])
        pred = predicted_actions[:, :horizon, :action_dim].to(torch.float32)
        target = target_actions[:, :horizon, :action_dim].to(torch.float32)
        diff = pred - target
        return {
            "action_mse": diff.square().mean().item(),
            "action_mae": diff.abs().mean().item(),
        }

    def run_eval(self) -> dict[str, float]:
        """Run offline CFG eval and log flow/action reconstruction metrics."""
        if not self.eval_data_loaders:
            return {}

        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)

            self.model.eval()
            final_metrics: dict[str, float] = {}
            global_sums: dict[str, float] = {}
            global_count = 0
            action_eval_batches = int(self.cfg.data.get("eval_action_batches", 0))

            with torch.no_grad():
                for ds_name, loader in self.eval_data_loaders:
                    metric_sums: dict[str, float] = {}
                    metric_counts: dict[str, float] = {}
                    for batch_idx, (observation, actions, advantage) in enumerate(
                        loader
                    ):
                        observation, actions, advantage = self._prepare_cfg_batch(
                            observation,
                            actions,
                            advantage,
                        )
                        batch_size = int(actions.shape[0])

                        with self.amp_context:
                            loss, metrics_data = self.model(
                                data={
                                    "observation": observation,
                                    "actions": actions,
                                    "advantage": advantage,
                                },
                            )
                            loss = loss.mean()

                        batch_metrics = {"loss": loss.detach().item()}

                        if action_eval_batches <= 0 or batch_idx < action_eval_batches:
                            sampled = self.model.sample_actions(observation)
                            batch_metrics.update(
                                self._action_error_metrics(
                                    sampled["actions"],
                                    actions,
                                )
                            )

                        for key, value in batch_metrics.items():
                            metric_sums[key] = (
                                metric_sums.get(key, 0.0) + float(value) * batch_size
                            )
                            metric_counts[key] = (
                                metric_counts.get(key, 0.0) + batch_size
                            )

                    if not metric_sums:
                        continue

                    reduce_payload = dict(metric_sums)
                    reduce_payload.update(
                        {f"{key}__count": value for key, value in metric_counts.items()}
                    )
                    reduced = all_reduce_dict(
                        reduce_payload,
                        op=torch.distributed.ReduceOp.SUM,
                    )
                    ds_metrics = {}
                    for key in metric_sums:
                        count = max(reduced.get(f"{key}__count", 0.0), 1.0)
                        ds_metrics[key] = reduced.get(key, 0.0) / count
                    for key, value in ds_metrics.items():
                        final_metrics[f"{ds_name}/{key}"] = value
                        global_sums[key] = global_sums.get(key, 0.0) + value
                    global_count += 1

            if global_count > 0:
                for key, value in global_sums.items():
                    final_metrics[key] = value / global_count

            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.offload_param_and_grad()

            return final_metrics

    def set_global_step(self, global_step):
        self.global_step = global_step

        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

        loader_len = len(self.data_loader)
        if loader_len == 0:
            return

        grad_accum = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        steps_per_epoch = max(1, loader_len // grad_accum)
        new_epoch = global_step // steps_per_epoch

        current_epoch = getattr(self, "_current_epoch", -1)
        if current_epoch != new_epoch:
            self._current_epoch = new_epoch
            self._data_epoch = new_epoch
            self._data_iter_offset = 0
            self.data_loader.set_epoch(new_epoch)
            self.data_iter = iter(self.data_loader)
