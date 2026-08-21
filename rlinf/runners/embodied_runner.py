# Copyright 2025 The RLinf Authors.
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

import logging
import json
import os
import queue
import threading
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Optional, Union
import numpy as np
import torch

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor",
            "EmbodiedNFTFSDPPolicy",
            "EmbodiedSACFSDPPolicy",
            "AsyncEmbodiedSACFSDPPolicy",
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        reward: Optional["EmbodiedRewardWorker"] = None,
        critic=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.cfg.algorithm.init_emb_dir is not None:
            data = np.load(self.cfg.algorithm.init_emb_dir)
            self.init_emb = data[list(data.keys())[0]]
        else:
            self.init_emb = np.zeros(1408)
        self.emb_threshold = self.cfg.algorithm.init_emb_threshold
        self.curriculum_delta = self.cfg.algorithm.curriculum_delta
        self.need_curriculum = bool(
            self.cfg.algorithm.get("init_emb_dir", None) is not None
            and self.cfg.algorithm.get("emb_buffer_dir", None) is not None
        )
        self.train_set_suc_rate = 0
        self.scenario_curriculum_cfg = self._get_scenario_curriculum_cfg()
        self.scenario_curriculum_enabled = bool(
            self.scenario_curriculum_cfg.get("enabled", False)
        )
        self.scenario_curriculum_stage_index = int(
            self.scenario_curriculum_cfg.get("initial_stage_index", 0) or 0
        )
        self.scenario_curriculum_success_threshold = float(
            self.scenario_curriculum_cfg.get("success_threshold", 0.7)
        )
        self.scenario_curriculum_success_thresholds = [
            float(threshold)
            for threshold in self.scenario_curriculum_cfg.get(
                "success_thresholds", []
            )
        ]
        self.scenario_curriculum_stable_steps_required = int(
            self.scenario_curriculum_cfg.get("stable_steps", 10)
        )
        self.scenario_curriculum_require_consecutive_success = bool(
            self.scenario_curriculum_cfg.get("require_consecutive_success", True)
        )
        self.scenario_curriculum_stable_success_steps = 0
        self.scenario_curriculum_stage_steps = 0
        self.scenario_curriculum_thresholds = list(
            self.scenario_curriculum_cfg.get("thresholds", [])
        )
        self.scenario_curriculum_promotion_cfg = dict(
            self.scenario_curriculum_cfg.get("promotion", {}) or {}
        )
        self.scenario_curriculum_promotion_mode = (
            self.scenario_curriculum_promotion_cfg.get("mode", "legacy")
        )
        self.scenario_curriculum_window = int(
            self.scenario_curriculum_promotion_cfg.get("window", 1) or 1
        )
        self.scenario_curriculum_min_steps_per_stage = int(
            self.scenario_curriculum_promotion_cfg.get("min_steps_per_stage", 0) or 0
        )
        self.scenario_curriculum_max_steps_per_stage = int(
            self.scenario_curriculum_promotion_cfg.get("max_steps_per_stage", 0) or 0
        )
        self.scenario_curriculum_success_history = deque(
            maxlen=max(self.scenario_curriculum_window, 1)
        )
        self.scenario_curriculum_new_success_history = deque(
            maxlen=max(self.scenario_curriculum_window, 1)
        )
        self.scenario_curriculum_old_success_history = deque(
            maxlen=max(self.scenario_curriculum_window, 1)
        )
        self.scenario_curriculum_cumulative_success_history = deque(
            maxlen=max(self.scenario_curriculum_window, 1)
        )
        self.scenario_curriculum_stage_groups = self._load_scenario_curriculum_groups()
        self.scenario_curriculum_state = {}
        if self.reward is not None:
            self.reward_channel = Channel.create("Reward")
        else:
            self.reward_channel = None

        # this timer checks if we should stop training
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

        # Async logging setup
        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()


    def _get_scenario_curriculum_cfg(self):
        try:
            scenario_reset_cfg = self.cfg.env.train.init_params.scenario_reset
        except Exception:
            return {}
        curriculum_cfg = scenario_reset_cfg.get("curriculum", {})
        return curriculum_cfg or {}

    def _load_scenario_curriculum_groups(self):
        manifest_file = self.scenario_curriculum_cfg.get("stage_manifest_file", None)
        if not manifest_file:
            return []
        try:
            with open(str(manifest_file), "r", encoding="utf-8") as fp:
                manifest = json.load(fp)
        except FileNotFoundError:
            return []

        groups = []
        previous_cumulative_ids: set[int] = set()
        for raw_stage in manifest.get("stages", []) or []:
            cumulative_ids = {
                int(scenario_id) for scenario_id in raw_stage.get("cumulative_ids", [])
            }
            incremental_ids = {
                int(scenario_id) for scenario_id in raw_stage.get("incremental_ids", [])
            }
            if not incremental_ids:
                incremental_ids = cumulative_ids - previous_cumulative_ids
            groups.append(
                {
                    "cumulative_ids": cumulative_ids,
                    "old_ids": set(previous_cumulative_ids),
                    "new_ids": incremental_ids,
                }
            )
            previous_cumulative_ids = set(cumulative_ids)
        return groups

    def _sync_scenario_curriculum_stage(self):
        if not self.scenario_curriculum_enabled:
            return []
        result = self.env.set_scenario_curriculum_stage(
            self.scenario_curriculum_stage_index
        ).wait()
        states = self._update_scenario_curriculum_state(result)
        if not states:
            raise RuntimeError(
                "scenario curriculum is enabled, but no EnvWorker returned an "
                "enabled scenario scheduler state"
            )
        return result

    def _sync_scenario_curriculum_progress(self):
        if not self.scenario_curriculum_enabled:
            return []
        result = self.env.set_scenario_curriculum_progress(
            self.scenario_curriculum_stage_index,
            self.scenario_curriculum_stage_steps,
        ).wait()
        self._update_scenario_curriculum_state(result)
        return result

    def _update_scenario_curriculum_state(self, worker_results):
        states = []
        for worker_result in worker_results or []:
            if not isinstance(worker_result, dict):
                continue
            for state in worker_result.get("states", []) or []:
                if isinstance(state, dict) and state.get("enabled", False):
                    states.append(state)
        if states:
            self.scenario_curriculum_state = states[0]
        return states

    def _reset_scenario_curriculum_stage_tracking(self):
        self.scenario_curriculum_stage_steps = 0
        self.scenario_curriculum_stable_success_steps = 0
        self.scenario_curriculum_success_history.clear()
        self.scenario_curriculum_new_success_history.clear()
        self.scenario_curriculum_old_success_history.clear()
        self.scenario_curriculum_cumulative_success_history.clear()

    def _current_scenario_curriculum_success_threshold(self):
        if (
            self.scenario_curriculum_stage_index
            < len(self.scenario_curriculum_success_thresholds)
        ):
            return self.scenario_curriculum_success_thresholds[
                self.scenario_curriculum_stage_index
            ]
        return self.scenario_curriculum_success_threshold

    def _promotion_threshold(self, key, fallback):
        values = self.scenario_curriculum_promotion_cfg.get(key, [])
        if (
            isinstance(values, (list, tuple))
            and self.scenario_curriculum_stage_index < len(values)
        ):
            return float(values[self.scenario_curriculum_stage_index])
        value = self.scenario_curriculum_promotion_cfg.get(key, None)
        if isinstance(value, (int, float)):
            return float(value)
        return float(fallback)

    def _history_mean(self, history):
        if not history:
            return None
        return float(sum(history) / len(history))

    def _compute_scenario_group_metrics(self, env_results_list):
        if (
            not self.scenario_curriculum_enabled
            or not self.scenario_curriculum_stage_groups
            or self.scenario_curriculum_stage_index
            >= len(self.scenario_curriculum_stage_groups)
        ):
            return {}

        scenario_id_tensors = []
        success_tensors = []
        for env_result in env_results_list or []:
            if "scenario_id" not in env_result or "success_once" not in env_result:
                continue
            scenario_id_tensors.append(
                env_result["scenario_id"].detach().cpu().reshape(-1).to(torch.int64)
            )
            success_tensors.append(
                env_result["success_once"].detach().cpu().reshape(-1).float()
            )
        if not scenario_id_tensors or not success_tensors:
            return {}

        scenario_ids = torch.cat(scenario_id_tensors, dim=0)
        successes = torch.cat(success_tensors, dim=0)
        groups = self.scenario_curriculum_stage_groups[
            self.scenario_curriculum_stage_index
        ]

        def group_stats(group_ids):
            if not group_ids:
                return None, 0.0
            mask_values = [int(scenario_id.item()) in group_ids for scenario_id in scenario_ids]
            mask = torch.tensor(mask_values, dtype=torch.bool)
            count = int(mask.sum().item())
            if count <= 0:
                return None, 0.0
            return float(successes[mask].mean().item()), float(count)

        cumulative_success, cumulative_count = group_stats(groups["cumulative_ids"])
        old_success, old_count = group_stats(groups["old_ids"])
        new_success, new_count = group_stats(groups["new_ids"])

        metrics = {
            "curriculum/cumulative_count": cumulative_count,
            "curriculum/old_cumulative_count": old_count,
            "curriculum/new_increment_count": new_count,
        }
        if cumulative_success is not None:
            metrics["curriculum/cumulative_success"] = cumulative_success
        if old_success is not None:
            metrics["curriculum/old_cumulative_success"] = old_success
        if new_success is not None:
            metrics["curriculum/new_increment_success"] = new_success
        return metrics

    def _scenario_curriculum_metrics(
        self,
        train_success=None,
        advanced=False,
        grouped_metrics=None,
        can_advance=False,
    ):
        if not self.scenario_curriculum_enabled:
            return {}
        state = self.scenario_curriculum_state or {}
        threshold = state.get("threshold", None)
        threshold_value = -1.0 if threshold is None else float(threshold)
        metrics = {
            "curriculum/stage_index": float(self.scenario_curriculum_stage_index),
            "curriculum/threshold": threshold_value,
            "curriculum/is_all_scenarios": float(threshold is None),
            "curriculum/allowed_scenarios": float(
                state.get("allowed_scenarios", 0) or 0
            ),
            "curriculum/total_scenarios": float(
                state.get("total_scenarios", 0) or 0
            ),
            "curriculum/stage_steps": float(self.scenario_curriculum_stage_steps),
            "curriculum/stable_success_steps": float(
                self.scenario_curriculum_stable_success_steps
            ),
            "curriculum/success_hits": float(
                self.scenario_curriculum_stable_success_steps
            ),
            "curriculum/required_success_hits": float(
                self.scenario_curriculum_stable_steps_required
            ),
            "curriculum/success_threshold": float(
                self._current_scenario_curriculum_success_threshold()
            ),
            "curriculum/success_window_mean": float(
                self._history_mean(self.scenario_curriculum_success_history) or 0.0
            ),
            "curriculum/new_success_window_mean": float(
                self._history_mean(self.scenario_curriculum_new_success_history) or 0.0
            ),
            "curriculum/old_success_window_mean": float(
                self._history_mean(self.scenario_curriculum_old_success_history) or 0.0
            ),
            "curriculum/cumulative_success_window_mean": float(
                self._history_mean(
                    self.scenario_curriculum_cumulative_success_history
                )
                or 0.0
            ),
            "curriculum/new_success_threshold": float(
                self._promotion_threshold(
                    "new_success_thresholds",
                    self._current_scenario_curriculum_success_threshold(),
                )
            ),
            "curriculum/cumulative_floor": float(
                self._promotion_threshold("cumulative_floor", 0.0)
            ),
            "curriculum/min_steps_per_stage": float(
                self.scenario_curriculum_min_steps_per_stage
            ),
            "curriculum/max_steps_per_stage": float(
                self.scenario_curriculum_max_steps_per_stage
            ),
            "curriculum/can_advance": float(bool(can_advance)),
            "curriculum/mix_old_ratio": float(state.get("mix_old_ratio", 0.0) or 0.0),
            "curriculum/mix_new_ratio": float(state.get("mix_new_ratio", 0.0) or 0.0),
            "curriculum/require_consecutive_success": float(
                self.scenario_curriculum_require_consecutive_success
            ),
            "curriculum/advanced": float(bool(advanced)),
        }
        if train_success is not None:
            metrics["curriculum/train_success_current_stage"] = float(train_success)
        if grouped_metrics:
            metrics.update(grouped_metrics)
        return metrics

    def _maybe_update_scenario_curriculum(self, train_success, grouped_metrics=None):
        if not self.scenario_curriculum_enabled:
            return {}
        if train_success is None:
            raise KeyError(
                "scenario curriculum requires env/success_once in train metrics"
            )
        if hasattr(train_success, "item"):
            train_success = train_success.item()
        train_success = float(train_success)
        success_threshold = self._current_scenario_curriculum_success_threshold()
        grouped_metrics = grouped_metrics or {}
        self.scenario_curriculum_stage_steps += 1

        if self.scenario_curriculum_promotion_mode == "grouped_moving_average":
            cumulative_success = float(
                grouped_metrics.get("curriculum/cumulative_success", train_success)
            )
            new_success = grouped_metrics.get("curriculum/new_increment_success", None)
            old_success = grouped_metrics.get("curriculum/old_cumulative_success", None)

            self.scenario_curriculum_success_history.append(train_success)
            self.scenario_curriculum_cumulative_success_history.append(
                cumulative_success
            )
            if new_success is not None:
                self.scenario_curriculum_new_success_history.append(float(new_success))
            if old_success is not None:
                self.scenario_curriculum_old_success_history.append(float(old_success))

            cumulative_mean = self._history_mean(
                self.scenario_curriculum_cumulative_success_history
            )
            new_mean = self._history_mean(self.scenario_curriculum_new_success_history)
            max_stage_index = max(len(self.scenario_curriculum_thresholds) - 1, 0)

            if self.scenario_curriculum_stage_index == 0:
                stage_threshold = self._promotion_threshold(
                    "cumulative_success_thresholds", success_threshold
                )
                metric_ready = (
                    cumulative_mean is not None and cumulative_mean >= stage_threshold
                )
            else:
                new_threshold = self._promotion_threshold(
                    "new_success_thresholds", success_threshold
                )
                cumulative_floor = self._promotion_threshold("cumulative_floor", 0.0)
                metric_ready = (
                    new_mean is not None
                    and new_mean >= new_threshold
                    and cumulative_mean is not None
                    and cumulative_mean >= cumulative_floor
                )

            can_advance = (
                self.scenario_curriculum_stage_steps
                >= self.scenario_curriculum_min_steps_per_stage
                and metric_ready
                and self.scenario_curriculum_stage_index < max_stage_index
            )
            force_advance = (
                self.scenario_curriculum_max_steps_per_stage > 0
                and self.scenario_curriculum_stage_steps
                >= self.scenario_curriculum_max_steps_per_stage
                and self.scenario_curriculum_stage_index < max_stage_index
            )
            advanced = False
            if can_advance or force_advance:
                previous_stage_index = self.scenario_curriculum_stage_index
                self.scenario_curriculum_stage_index += 1
                self._reset_scenario_curriculum_stage_tracking()
                self._sync_scenario_curriculum_stage()
                advanced = True
                reason = (
                    "max_steps_per_stage"
                    if force_advance and not can_advance
                    else "metrics"
                )
                self.logger.info(
                    "Advanced scenario curriculum to stage "
                    f"{self.scenario_curriculum_stage_index}: "
                    f"{self.scenario_curriculum_state}; "
                    f"reason={reason}, previous_stage={previous_stage_index}"
                )
            return self._scenario_curriculum_metrics(
                train_success,
                advanced=advanced,
                grouped_metrics=grouped_metrics,
                can_advance=can_advance or force_advance,
            )

        if train_success >= success_threshold:
            self.scenario_curriculum_stable_success_steps += 1
        elif self.scenario_curriculum_require_consecutive_success:
            self.scenario_curriculum_stable_success_steps = 0

        advanced = False
        max_stage_index = max(len(self.scenario_curriculum_thresholds) - 1, 0)
        if (
            (
                self.scenario_curriculum_stable_success_steps
                >= self.scenario_curriculum_stable_steps_required
            )
            or (
                self.scenario_curriculum_max_steps_per_stage > 0
                and self.scenario_curriculum_stage_steps
                >= self.scenario_curriculum_max_steps_per_stage
            )
        ) and self.scenario_curriculum_stage_index < max_stage_index:
            reason = (
                "metrics"
                if self.scenario_curriculum_stable_success_steps
                >= self.scenario_curriculum_stable_steps_required
                else "max_steps_per_stage"
            )
            previous_stage_index = self.scenario_curriculum_stage_index
            self.scenario_curriculum_stage_index += 1
            self._reset_scenario_curriculum_stage_tracking()
            self._sync_scenario_curriculum_stage()
            advanced = True
            self.logger.info(
                "Advanced scenario curriculum to stage "
                f"{self.scenario_curriculum_stage_index}: "
                f"{self.scenario_curriculum_state}; "
                f"reason={reason}, previous_stage={previous_stage_index}"
            )

        return self._scenario_curriculum_metrics(
            train_success,
            advanced=advanced,
            grouped_metrics=grouped_metrics,
            can_advance=advanced,
        )

    def _log_worker(self):
        """Background thread for processing log messages."""
        while not self.stop_logging:
            try:
                # Wait for log message with timeout
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logging error: {e}")
                continue

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        """Async version that puts table printing in queue."""
        self.log_queue.put(
            (print_metrics_table, (step, total_steps, start_time, metrics, start_step))
        )

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()
        if self.reward is not None:
            self.reward.init_worker().wait()

        rollout_handle.wait()
        env_handle.wait()
        self.actor.init_worker().wait()

        if self.scenario_curriculum_enabled:
            self._sync_scenario_curriculum_stage()
            self.logger.info(
                "Initialized scenario curriculum: "
                f"{self.scenario_curriculum_state}"
            )

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [
            {
                key: value
                for key, value in results.items()
                if key not in {"scenario_id"}
            }
            for results in env_results
            if results is not None
        ]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | None) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    def threshold2envid(self, emb_threshold):
        """
        function: givin a thre, cal env_id, and write to a local file
        buffer: csv(env_id, distance_to_init_emb)
        """
        emb_buffer_dir = self.cfg.algorithm.emb_buffer_dir
        # 从 csv 中小于阈值的 env_id 记录到一个文件下 (isaac 那边要做的是从这个文件的 id 中随机挑一个去初始化)
        pass

    def run(self):
        start_step = self.global_step
        start_time = time.time()
        if self.need_curriculum:
            self.rollout.set_global_emb(self.init_emb)
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            if self.train_set_suc_rate >= 0.7:
                self.emb_threshold += self.curriculum_delta
            if self.need_curriculum:
                # self.rollout.set_global_threshold(self.emb_threshold)
                self.threshold2envid(self.emb_threshold)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                if self.scenario_curriculum_enabled:
                    self._sync_scenario_curriculum_progress()
                with self.timer("generate_rollouts"):
                    env_handle: Handle = self.env.interact(
                        input_channel=self.env_channel,
                        rollout_channel=self.rollout_channel,
                        reward_channel=self.reward_channel,
                        actor_channel=self.actor_channel,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    if self.reward is not None:
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.reward_channel,
                            output_channel=self.env_channel,
                        )
                    self.actor.recv_rollout_trajectories(
                        input_channel=self.actor_channel
                    ).wait()
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                # actor training.
                actor_training_handle: Handle = self.actor.run_training()

                actor_training_metrics = actor_training_handle.wait()

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                eval_metrics = {}
                if run_val:
                    with self.timer("eval"):
                        self.update_rollout_weights()
                        eval_metrics = self.evaluate()
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=_step)

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            env_time_metrics, env_time_metrics_per_rank = env_handle.consume_durations(
                return_per_rank=True
            )
            rollout_time_metrics, rollout_time_metrics_per_rank = (
                rollout_handle.consume_durations(return_per_rank=True)
            )
            actor_time_metrics, actor_time_metrics_per_rank = (
                actor_training_handle.consume_durations(return_per_rank=True)
            )
            time_metrics.update(
                {f"time/env/{k}": v for k, v in env_time_metrics.items()}
            )
            time_metrics.update(
                {f"time/rollout/{k}": v for k, v in rollout_time_metrics.items()}
            )
            time_metrics.update(
                {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
            )
            if self.reward is not None:
                reward_time_metrics, reward_time_metrics_per_rank = (
                    reward_handle.consume_durations(return_per_rank=True)
                )
                time_metrics.update(
                    {f"time/reward/{k}": v for k, v in reward_time_metrics.items()}
                )

            env_results = env_handle.wait()
            env_results_list = [
                results for results in env_results if results is not None
            ]
            curriculum_grouped_metrics = self._compute_scenario_group_metrics(
                env_results_list
            )
            env_metric_results_list = [
                {
                    key: value
                    for key, value in results.items()
                    if key not in {"scenario_id"}
                }
                for results in env_results_list
            ]
            env_metrics = compute_evaluate_metrics(env_metric_results_list)
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            ranked_env_results = [
                {
                    "rank": rank,
                    "env": {
                        key: value
                        for key, value in rank_metrics.items()
                        if key not in {"scenario_id"}
                    },
                }
                for rank, rank_metrics in enumerate(env_results)
                if rank_metrics is not None
            ]
            _, env_metrics_per_rank = self._process_ranked_eval_results(
                ranked_env_results, metric_field="env"
            )

            rollout_metrics = {
                f"rollout/{k}": v
                for k, v in self._aggregate_numeric_metrics(
                    actor_rollout_metrics
                ).items()
            }
            training_metrics = {
                f"train/{k}": v
                for k, v in self._aggregate_numeric_metrics(
                    actor_training_metrics
                ).items()
            }
            curriculum_metrics = self._maybe_update_scenario_curriculum(
                env_metrics.get("env/success_once"),
                grouped_metrics=curriculum_grouped_metrics,
            )

            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)
            if curriculum_metrics:
                self.metric_logger.log(curriculum_metrics, _step)
            self._log_ranked_metrics(
                metrics_list=actor_rollout_metrics,
                step=_step,
                prefix="rollout",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_training_metrics,
                step=_step,
                prefix="train",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_time_metrics_per_rank,
                step=_step,
                prefix="time/actor",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=rollout_time_metrics_per_rank,
                step=_step,
                prefix="time/rollout",
                worker_group_name=self.rollout.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_time_metrics_per_rank,
                step=_step,
                prefix="time/env",
                worker_group_name=self.env.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_metrics_per_rank,
                step=_step,
                prefix="env",
                worker_group_name=self.env.worker_group_name,
            )
            if self.reward is not None:
                self._log_ranked_metrics(
                    metrics_list=reward_time_metrics_per_rank,
                    step=_step,
                    prefix="time/reward",
                    worker_group_name=self.reward.worker_group_name,
                )

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)
            logging_metrics.update(curriculum_metrics)

            self.train_set_suc_rate = env_metrics["env/success_once"]
            print(f"self.train_set_suc_rate: {self.train_set_suc_rate}")

            self.print_metrics_table_async(
                _step, self.max_steps, start_time, logging_metrics, start_step
            )

        self.metric_logger.finish()

        # Stop logging thread
        self.stop_logging = True
        self.log_queue.join()  # Wait for all queued logs to be processed
        self.log_thread.join(timeout=1.0)

    def _save_checkpoint(self):
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
