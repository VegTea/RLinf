import json
import random
from pathlib import Path


class ScenarioScheduler:
    def __init__(
        self,
        scenario_ids,
        mode="sequential",
        fixed_ids=None,
        external_group_a_ids=None,
        external_group_b_ids=None,
        ratio_a=0.8,
        loop=True,
        seed=0,
        scenario_records=None,
        curriculum=None,
    ):
        self.scenario_ids = [str(scenario_id) for scenario_id in scenario_ids]
        self.mode = mode
        self.fixed_ids = [str(scenario_id) for scenario_id in (fixed_ids or [])]
        self.external_group_a_ids = [
            str(scenario_id) for scenario_id in (external_group_a_ids or [])
        ]
        self.external_group_b_ids = [
            str(scenario_id) for scenario_id in (external_group_b_ids or [])
        ]
        self.ratio_a = float(ratio_a)
        self.loop = loop
        self.seed = int(seed)

        self.curriculum = dict(curriculum or {})
        self.curriculum_enabled = bool(self.curriculum.get("enabled", False))
        self.curriculum_stage_index = int(
            self.curriculum.get("initial_stage_index", 0) or 0
        )
        self.curriculum_thresholds = self._normalize_thresholds(
            self.curriculum.get("thresholds", [])
        )
        self.curriculum_stage_manifest_file = self.curriculum.get(
            "stage_manifest_file"
        )
        self.curriculum_stage_ids = []
        self.curriculum_stage_step = 0
        self.curriculum_sampling_cfg = dict(self.curriculum.get("sampling", {}) or {})
        self.curriculum_mix_schedule = list(
            self.curriculum_sampling_cfg.get("mix_schedule", []) or []
        )
        self.curriculum_distance_key = self.curriculum.get(
            "distance_key", "cosine_distance"
        )
        self.scenario_distance_by_id = {}
        if self.curriculum_enabled and scenario_records is not None:
            self.scenario_distance_by_id = self._build_distance_map(scenario_records)
        if self.curriculum_enabled and self.curriculum_stage_manifest_file:
            self.curriculum_stage_ids = self._load_stage_manifest(
                self.curriculum_stage_manifest_file
            )
            if not self.curriculum_thresholds:
                self.curriculum_thresholds = [
                    stage.get("threshold") for stage in self.curriculum_stage_ids
                ]
        self.allowed_ids = list(self.scenario_ids)

        if self.mode == "by_id" and not self.fixed_ids:
            raise ValueError("scenario_reset.mode=by_id requires fixed_ids")
        if self.mode == "external":
            if not self.external_group_a_ids and not self.external_group_b_ids:
                raise ValueError(
                    "scenario_reset.mode=external requires group_a_ids or group_b_ids"
                )
        if self.curriculum_enabled:
            if not self.curriculum_thresholds and not self.curriculum_stage_ids:
                raise ValueError(
                    "scenario_reset.curriculum.thresholds or stage_manifest_file "
                    "must not be empty"
                )
            self.set_curriculum_stage(self.curriculum_stage_index)

    def _normalize_thresholds(self, thresholds):
        normalized = []
        for threshold in thresholds or []:
            if threshold is None:
                normalized.append(None)
            elif isinstance(threshold, str) and threshold.lower() in {"none", "null", "all"}:
                normalized.append(None)
            else:
                normalized.append(float(threshold))
        return normalized

    def _build_distance_map(self, scenario_records):
        distance_by_id = {}
        for record in scenario_records:
            scenario_id = str(record["id"])
            if self.curriculum_distance_key not in record:
                raise ValueError(
                    f"Scenario {scenario_id} is missing curriculum distance key "
                    f"{self.curriculum_distance_key!r}"
                )
            distance_by_id[scenario_id] = float(record[self.curriculum_distance_key])
        return distance_by_id

    def _load_stage_manifest(self, manifest_file):
        manifest_path = Path(manifest_file)
        with manifest_path.open("r", encoding="utf-8") as fp:
            manifest = json.load(fp)

        raw_stages = manifest.get("stages")
        if not raw_stages:
            raise ValueError(
                f"Curriculum stage manifest has no stages: {manifest_file}"
            )

        scenario_id_set = set(self.scenario_ids)
        stages = []
        for expected_index, raw_stage in enumerate(raw_stages):
            stage_index = int(raw_stage.get("stage_index", expected_index))
            if stage_index != expected_index:
                raise ValueError(
                    "Curriculum stage manifest must use contiguous stage_index "
                    f"values starting at 0; expected {expected_index}, got {stage_index}"
                )

            cumulative_ids = [
                str(scenario_id)
                for scenario_id in raw_stage.get("cumulative_ids", [])
            ]
            if not cumulative_ids:
                raise ValueError(
                    f"Curriculum stage {stage_index} in {manifest_file} has no ids"
                )

            missing_ids = [
                scenario_id
                for scenario_id in cumulative_ids
                if scenario_id not in scenario_id_set
            ]
            if missing_ids:
                preview = ", ".join(missing_ids[:10])
                raise ValueError(
                    f"Curriculum stage {stage_index} contains ids not present in "
                    f"scenario_file: {preview}"
                )

            threshold = raw_stage.get("threshold")
            if threshold is None:
                threshold = raw_stage.get("max_distance")
            stages.append(
                {
                    "stage_index": stage_index,
                    "cumulative_ids": cumulative_ids,
                    "incremental_ids": [
                        str(scenario_id)
                        for scenario_id in raw_stage.get("incremental_ids", [])
                    ],
                    "threshold": None if threshold is None else float(threshold),
                    "incremental_count": int(
                        raw_stage.get("incremental_count", len(cumulative_ids))
                    ),
                }
            )
        return stages

    def set_curriculum_stage(self, stage_index):
        if not self.curriculum_enabled:
            return self.get_curriculum_state()

        stage_index = int(stage_index)
        num_stages = (
            len(self.curriculum_stage_ids)
            if self.curriculum_stage_ids
            else len(self.curriculum_thresholds)
        )
        if stage_index < 0 or stage_index >= num_stages:
            raise IndexError(
                f"Curriculum stage index {stage_index} is out of range for "
                f"{num_stages} stages"
            )

        if self.curriculum_stage_ids:
            allowed_ids = list(
                self.curriculum_stage_ids[stage_index]["cumulative_ids"]
            )
        elif not self.scenario_distance_by_id:
            raise ValueError(
                "scenario_reset.curriculum requires scenario records with distances"
            )
        else:
            threshold = self.curriculum_thresholds[stage_index]
            if threshold is None:
                allowed_ids = list(self.scenario_ids)
            else:
                allowed_ids = [
                    scenario_id
                    for scenario_id in self.scenario_ids
                    if self.scenario_distance_by_id[scenario_id] <= threshold
                ]
        if not allowed_ids:
            raise ValueError(
                "Curriculum stage produced no scenarios: "
                f"stage_index={stage_index}, "
                f"distance_key={self.curriculum_distance_key}"
            )

        self.curriculum_stage_index = stage_index
        self.curriculum_stage_step = 0
        self.allowed_ids = allowed_ids
        return self.get_curriculum_state()

    def set_curriculum_progress(self, stage_index=None, stage_step=None):
        if not self.curriculum_enabled:
            return self.get_curriculum_state()
        if stage_index is not None and int(stage_index) != self.curriculum_stage_index:
            self.set_curriculum_stage(int(stage_index))
        if stage_step is not None:
            self.curriculum_stage_step = max(int(stage_step), 0)
        return self.get_curriculum_state()

    def get_curriculum_state(self):
        threshold = None
        stage_source = "threshold"
        if self.curriculum_stage_ids:
            stage_source = "manifest"
            threshold = self.curriculum_stage_ids[
                self.curriculum_stage_index
            ].get("threshold")
        elif self.curriculum_thresholds:
            threshold = self.curriculum_thresholds[self.curriculum_stage_index]
        return {
            "enabled": self.curriculum_enabled,
            "stage_index": self.curriculum_stage_index,
            "threshold": threshold,
            "allowed_scenarios": len(self.allowed_ids),
            "total_scenarios": len(self.scenario_ids),
            "distance_key": self.curriculum_distance_key,
            "stage_source": stage_source,
            "stage_step": int(self.curriculum_stage_step),
            "old_scenarios": len(self._current_old_ids()),
            "new_scenarios": len(self._current_new_ids()),
            "mix_old_ratio": float(self._current_mix_ratios()[0]),
            "mix_new_ratio": float(self._current_mix_ratios()[1]),
            "is_all_scenarios": threshold is None,
        }

    def _active_ids(self):
        return self.allowed_ids if self.curriculum_enabled else self.scenario_ids

    def _current_old_ids(self):
        if not self.curriculum_stage_ids or self.curriculum_stage_index <= 0:
            return []
        return list(
            self.curriculum_stage_ids[self.curriculum_stage_index - 1][
                "cumulative_ids"
            ]
        )

    def _current_new_ids(self):
        if not self.curriculum_stage_ids:
            return []
        stage = self.curriculum_stage_ids[self.curriculum_stage_index]
        incremental_ids = list(stage.get("incremental_ids", []) or [])
        if incremental_ids:
            return incremental_ids
        old_ids = set(self._current_old_ids())
        return [
            scenario_id
            for scenario_id in stage.get("cumulative_ids", [])
            if scenario_id not in old_ids
        ]

    def _current_mix_ratios(self):
        if (
            not self.curriculum_stage_ids
            or self.curriculum_stage_index <= 0
            or not self.curriculum_mix_schedule
        ):
            return 0.0, 0.0

        elapsed = int(self.curriculum_stage_step)
        boundary = 0
        for entry in self.curriculum_mix_schedule:
            steps = int(entry.get("steps", 0) or 0)
            if steps <= 0:
                continue
            boundary += steps
            if elapsed < boundary:
                old_ratio = float(entry.get("old_ratio", 0.0) or 0.0)
                new_ratio = float(entry.get("new_ratio", 0.0) or 0.0)
                total = old_ratio + new_ratio
                if total <= 0:
                    return 0.0, 0.0
                return old_ratio / total, new_ratio / total
        return 0.0, 0.0

    def _cycle_take(self, ids, start, count):
        if not ids:
            return []
        if not self.loop and start + count > len(ids):
            raise IndexError("Scenario ids exhausted and loop is disabled")
        return [ids[(start + idx) % len(ids)] for idx in range(count)]

    def _sample_random_batch(self, global_batch_size, batch_index):
        ids = self._active_ids()
        if not ids:
            raise ValueError("scenario_reset.mode=random requires scenario ids")

        rng = random.Random(self.seed + batch_index)
        old_ratio, new_ratio = self._current_mix_ratios()
        old_ids = self._current_old_ids()
        new_ids = self._current_new_ids()
        if old_ratio > 0 and new_ratio > 0 and old_ids and new_ids:
            new_count = int(round(global_batch_size * new_ratio))
            new_count = min(max(new_count, 0), global_batch_size)
            old_count = global_batch_size - new_count
            if self.loop:
                sampled = (
                    [rng.choice(old_ids) for _ in range(old_count)]
                    + [rng.choice(new_ids) for _ in range(new_count)]
                )
            else:
                if old_count > len(old_ids) or new_count > len(new_ids):
                    raise IndexError(
                        "Mixed random scenario batch is larger than available ids "
                        "and loop is disabled"
                    )
                sampled = rng.sample(old_ids, old_count) + rng.sample(new_ids, new_count)
            rng.shuffle(sampled)
            return sampled

        if self.loop:
            return [rng.choice(ids) for _ in range(global_batch_size)]
        if global_batch_size > len(ids):
            raise IndexError(
                "Random scenario batch is larger than available ids and loop is disabled"
            )
        return rng.sample(ids, global_batch_size)

    def _sample_external_batch(self, global_batch_size, batch_index):
        count_a = int(round(global_batch_size * self.ratio_a))
        count_a = min(max(count_a, 0), global_batch_size)
        count_b = global_batch_size - count_a

        start_a = batch_index * max(count_a, 1)
        start_b = batch_index * max(count_b, 1)
        sampled_a = self._cycle_take(self.external_group_a_ids, start_a, count_a)
        sampled_b = self._cycle_take(self.external_group_b_ids, start_b, count_b)

        merged = sampled_a + sampled_b
        if not merged:
            raise ValueError("External scenario sampling produced an empty batch")

        rng = random.Random(self.seed + batch_index)
        rng.shuffle(merged)

        if len(merged) < global_batch_size:
            merged = self._cycle_take(merged, 0, global_batch_size)
        return merged

    def sample_global_batch(self, global_batch_size, batch_index):
        if self.mode == "sequential":
            return self._cycle_take(
                self._active_ids(),
                batch_index * global_batch_size,
                global_batch_size,
            )
        if self.mode == "by_id":
            return self._cycle_take(
                self.fixed_ids,
                batch_index * global_batch_size,
                global_batch_size,
            )
        if self.mode == "external":
            return self._sample_external_batch(global_batch_size, batch_index)
        if self.mode == "random":
            return self._sample_random_batch(global_batch_size, batch_index)
        raise ValueError(f"Unsupported scenario reset mode: {self.mode}")
