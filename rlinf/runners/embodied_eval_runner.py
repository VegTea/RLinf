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

import json
import os
import typing

import torch

from rlinf.envs.isaaclab.scenario_export import normalize_eval_scenarios
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def _concat_raw_eval_metrics(self, eval_metrics_list):
        raw_eval_metrics = {}
        for eval_metrics in eval_metrics_list:
            for key, value in eval_metrics.items():
                if isinstance(value, torch.Tensor):
                    raw_eval_metrics.setdefault(key, []).append(value.detach().cpu())

        for key, shards in raw_eval_metrics.items():
            raw_eval_metrics[key] = torch.cat(shards, dim=0).contiguous()

        return raw_eval_metrics

    def _save_eval_metrics(self, raw_eval_metrics, eval_metrics):
        log_path = self.cfg.runner.logger.log_path
        os.makedirs(log_path, exist_ok=True)
        save_path = os.path.join(log_path, "eval_metrics.pt")
        if self.cfg.env.eval.get("debug_init_cube_positions", False):
            self.logger.info(f"Raw eval metric keys: {sorted(raw_eval_metrics.keys())}")
            if "init_cube_positions" in raw_eval_metrics:
                self.logger.info(
                    "init_cube_positions shape: "
                    f"{tuple(raw_eval_metrics['init_cube_positions'].shape)}"
                )
                self.logger.info(
                    "init_cube_positions sample: "
                    f"{raw_eval_metrics['init_cube_positions'][:2].tolist()}"
                )
            else:
                self.logger.info("init_cube_positions is missing from raw eval metrics")
        torch.save(
            {
                "raw_metrics": raw_eval_metrics,
                "aggregated_metrics": eval_metrics,
            },
            save_path,
        )
        self.logger.info(f"Saved eval metrics to {save_path}")

    def _save_eval_manifest(self, raw_eval_metrics):
        manifest_path = self.cfg.runner.get("eval_manifest_path", None)
        scenario_output_path = self.cfg.runner.get(
            "eval_scenario_output_path", None
        )
        if not manifest_path and not scenario_output_path:
            return

        result_dir = self.cfg.env.eval.get("per_episode_result_dir", None)
        if not result_dir:
            raise ValueError(
                "runner.eval_manifest_path requires env.eval.per_episode_result_dir"
            )

        required_keys = (
            "worker_rank",
            "stage_id",
            "env_id",
            "episode_id",
        )
        missing_keys = [key for key in required_keys if key not in raw_eval_metrics]
        if missing_keys:
            raise RuntimeError(
                "Cannot build the per-environment eval manifest; missing raw "
                f"metric keys: {missing_keys}"
            )

        records = []
        seen_paths = set()
        num_rows = int(raw_eval_metrics["env_id"].numel())
        for idx in range(num_rows):
            worker_rank = int(raw_eval_metrics["worker_rank"][idx].item())
            stage_id = int(raw_eval_metrics["stage_id"][idx].item())
            env_id = int(raw_eval_metrics["env_id"][idx].item())
            episode_id = int(raw_eval_metrics["episode_id"][idx].item())
            result_path = os.path.join(
                str(result_dir),
                f"worker_{worker_rank:03d}_stage_{stage_id:02d}_"
                f"env_{env_id:03d}_episode_{episode_id:06d}.json",
            )
            if result_path in seen_paths:
                continue
            seen_paths.add(result_path)
            if not os.path.isfile(result_path):
                raise FileNotFoundError(
                    f"Per-environment eval result is missing: {result_path}"
                )
            with open(result_path, encoding="utf-8") as result_fp:
                record = json.load(result_fp)
            if manifest_path:
                video_path = record.get("video_path")
                if not video_path:
                    raise FileNotFoundError(
                        "Per-environment eval video is missing from result: "
                        f"{result_path}"
                    )
                if not os.path.isfile(video_path):
                    raise FileNotFoundError(
                        f"Per-environment eval video is missing: {video_path}"
                    )
            records.append(record)

        records.sort(
            key=lambda row: (
                row["worker_rank"],
                row["stage_id"],
                row["env_id"],
                row["episode_id"],
            )
        )
        if scenario_output_path:
            self._save_eval_scenarios(records, scenario_output_path)
        if not manifest_path:
            return

        video_base_dir = os.path.abspath(
            str(self.cfg.env.eval.video_cfg.video_base_dir)
        )
        successful_results = [
            os.path.relpath(record["video_path"], video_base_dir)
            for record in records
            if bool(record.get("policy_success", False))
        ]
        failed_results = [
            os.path.relpath(record["video_path"], video_base_dir)
            for record in records
            if not bool(record.get("policy_success", False))
        ]
        num_results = len(records)
        payload = {
            "num_results": num_results,
            "num_success": len(successful_results),
            "num_failed": len(failed_results),
            "success_rate": (
                len(successful_results) / num_results if num_results else 0.0
            ),
            "successful_results": successful_results,
            "failed_results": failed_results,
        }
        manifest_path = os.path.abspath(str(manifest_path))
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        tmp_path = f"{manifest_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as manifest_fp:
            json.dump(payload, manifest_fp, indent=2, ensure_ascii=False)
            manifest_fp.write("\n")
        os.replace(tmp_path, manifest_path)
        self.logger.info(f"Saved per-environment eval manifest to {manifest_path}")

    def _save_eval_throughput(self, raw_eval_metrics):
        throughput_path = self.cfg.runner.get("eval_throughput_path", None)
        if not throughput_path:
            return

        required_keys = ("success_once", "first_success_step", "episode_len")
        missing_keys = [key for key in required_keys if key not in raw_eval_metrics]
        if missing_keys:
            raise RuntimeError(
                "Cannot calculate eval success throughput; missing raw metric "
                f"keys: {missing_keys}"
            )

        successes = raw_eval_metrics["success_once"].reshape(-1).bool()
        first_success_steps = raw_eval_metrics["first_success_step"].reshape(-1)
        episode_lens = raw_eval_metrics["episode_len"].reshape(-1)
        if not (
            successes.numel()
            == first_success_steps.numel()
            == episode_lens.numel()
        ):
            raise RuntimeError(
                "Cannot calculate eval success throughput: metric lengths differ "
                f"(success_once={successes.numel()}, "
                f"first_success_step={first_success_steps.numel()}, "
                f"episode_len={episode_lens.numel()})"
            )
        if successes.numel() == 0:
            raise RuntimeError(
                "Cannot calculate eval success throughput from zero results"
            )
        if torch.any(successes & (first_success_steps <= 0)):
            raise RuntimeError(
                "Cannot calculate eval success throughput: a successful result "
                "has no positive first_success_step"
            )

        effective_steps = torch.where(
            successes,
            first_success_steps.to(torch.int64),
            episode_lens.to(torch.int64),
        )
        if torch.any(effective_steps <= 0):
            raise RuntimeError(
                "Cannot calculate eval success throughput: effective steps must "
                "all be positive"
            )

        num_results = int(successes.numel())
        num_success = int(successes.sum().item())
        successful_effective_steps = int(effective_steps[successes].sum().item())
        failed_effective_steps = int(effective_steps[~successes].sum().item())
        total_effective_steps = successful_effective_steps + failed_effective_steps
        success_throughput = num_success / total_effective_steps
        payload = {
            "definition": (
                "num_success / sum(first_success_step_if_success_else_episode_len)"
            ),
            "num_results": num_results,
            "num_success": num_success,
            "num_failed": num_results - num_success,
            "successful_effective_steps": successful_effective_steps,
            "failed_effective_steps": failed_effective_steps,
            "total_effective_steps": total_effective_steps,
            "success_throughput_per_step": success_throughput,
            "successes_per_1000_steps": success_throughput * 1000,
        }
        throughput_path = os.path.abspath(str(throughput_path))
        os.makedirs(os.path.dirname(throughput_path), exist_ok=True)
        tmp_path = f"{throughput_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as throughput_fp:
            json.dump(payload, throughput_fp, indent=2, ensure_ascii=False)
            throughput_fp.write("\n")
        os.replace(tmp_path, throughput_path)
        self.logger.info(f"Saved eval success throughput to {throughput_path}")

    def _save_eval_scenarios(self, records, scenario_output_path):
        expected_count = int(self.cfg.env.eval.total_num_envs)
        if len(records) != expected_count:
            raise RuntimeError(
                "Cannot save eval scenarios: expected "
                f"{expected_count} per-environment results, got {len(records)}"
            )

        scenario_records = []
        for index, result in enumerate(records):
            scenario_record = result.get("scenario_record")
            if not isinstance(scenario_record, dict):
                raise RuntimeError(
                    "Cannot save eval scenarios: result "
                    f"{index} has no captured scenario_record"
                )
            scenario_records.append(scenario_record)
        normalized = normalize_eval_scenarios(scenario_records)

        output_path = os.path.abspath(str(scenario_output_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tmp_path = f"{output_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as scenario_fp:
            for record in normalized:
                scenario_fp.write(
                    json.dumps(record, separators=(",", ":"), ensure_ascii=False)
                    + "\n"
                )
        os.replace(tmp_path, output_path)
        self.logger.info(
            f"Saved {len(normalized)} reloadable eval scenarios to {output_path}"
        )

    def _trajectory_record_cfg(self):
        return self.cfg.env.eval.get("trajectory_record_cfg", None)

    def _trajectory_record_enabled(self):
        cfg = self._trajectory_record_cfg()
        return cfg is not None and bool(getattr(cfg, "enabled", False))

    def _trajectory_record_stop_enabled(self):
        cfg = self._trajectory_record_cfg()
        return self._trajectory_record_enabled() and bool(
            getattr(cfg, "stop_when_targets_met", False)
        )

    def _trajectory_targets(self):
        cfg = self._trajectory_record_cfg()
        return (
            int(getattr(cfg, "target_success", 1000)),
            int(getattr(cfg, "target_fail", 1000)),
        )

    def _get_trajectory_record_counts(self):
        counts_list = self.env.get_eval_trajectory_record_counts().wait()
        aggregate = {
            "enabled": False,
            "success": 0,
            "fail": 0,
            "skipped_success": 0,
            "skipped_fail": 0,
        }
        for item in counts_list:
            worker_counts = item.get("aggregate", {}) if isinstance(item, dict) else {}
            aggregate["enabled"] = aggregate["enabled"] or bool(
                worker_counts.get("enabled", False)
            )
            for key in ("success", "fail", "skipped_success", "skipped_fail"):
                aggregate[key] = max(aggregate[key], int(worker_counts.get(key, 0)))
        return aggregate

    def _trajectory_targets_met(self, counts):
        target_success, target_fail = self._trajectory_targets()
        return (
            int(counts.get("success", 0)) >= target_success
            and int(counts.get("fail", 0)) >= target_fail
        )

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        rollout_handle.wait()
        env_handle.wait()

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
        eval_metrics_list = [results for results in env_results if results is not None]
        raw_eval_metrics = self._concat_raw_eval_metrics(eval_metrics_list)
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        self._save_eval_metrics(raw_eval_metrics, eval_metrics)
        self._save_eval_throughput(raw_eval_metrics)
        self._save_eval_manifest(raw_eval_metrics)
        return eval_metrics

    def run(self):
        max_rounds = 1
        if self._trajectory_record_stop_enabled():
            cfg = self._trajectory_record_cfg()
            max_rounds = int(getattr(cfg, "max_eval_rounds", 5000))

        for round_idx in range(max_rounds):
            eval_metrics = self.evaluate()
            eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
            self.logger.info(eval_metrics)
            self.metric_logger.log(step=round_idx, data=eval_metrics)

            if not self._trajectory_record_stop_enabled():
                break

            counts = self._get_trajectory_record_counts()
            target_success, target_fail = self._trajectory_targets()
            self.logger.info(
                "Eval trajectory recorder counts after round "
                f"{round_idx}: {counts}, targets: "
                f"success={target_success}, fail={target_fail}"
            )
            if self._trajectory_targets_met(counts):
                break
        else:
            counts = self._get_trajectory_record_counts()
            target_success, target_fail = self._trajectory_targets()
            raise RuntimeError(
                "Eval trajectory recorder did not reach requested quotas "
                f"within {max_rounds} rounds: counts={counts}, "
                f"targets success={target_success}, fail={target_fail}"
            )

        self.metric_logger.finish()
