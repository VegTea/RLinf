# Curriculum Success Rate Handoff

Last updated: 2026-07-30

This document summarizes the curriculum-learning comparison work, evaluation sources, success-rate statistics, and handoff notes for the IsaacLab stack-cube setting experiments.

## 1. Main Evaluation Models

The following three models were used most frequently in the comparisons:

1. **Base pi05 SFT Stack Cube**
   - Eval dir:
     `/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/logs/all_settings_exact_once_eval/20260727-13:12:52-single-ray-resumable-8x4090-RLinf-pi05-SFT-Stack-cube-isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum`
   - Full all-setting result: `1759 / 10025 = 17.55%`

2. **nearest100 global_step_400**
   - Eval dir:
     `/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/logs/all_settings_exact_once_eval/single-ray-resumable-8x4090-rlinf_nearest100_global_step_400_max450-isaaclab_franka_stack_cube_ppo_openpi_pi05_table_nearest100`
   - Full all-setting result: `3374 / 10025 = 33.66%`

3. **10stage200 mix50 max300 global_step_800**
   - Eval dir:
     `/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/logs/all_settings_exact_once_eval/single-ray-resumable-8x4090-rlinf_10stage200_mix50_max300_global_step_800_max450-isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200_mix50_max300`
   - Full all-setting result: `3054 / 10025 = 30.46%`

All three evaluation dirs contain `per_setting_results/*.json`. A setting is counted as successful when `success_rate > 0` or `num_success > 0`.

## 2. Distance And Stage Files

Important scenario/distance files:

- Original cosine/JEP-like distance:
  `/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/rlinf/assets_isaaclab/all_setting/combined_with_distance/Isaaclab_all_scenarios_10025_with_distance.jsonl`
  - Main key: `cosine_distance`

- I-JEPA/V-JEPA distance:
  `/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/rlinf/assets_isaaclab/all_setting/combined_with_ijepa_distance/Isaaclab_all_scenarios_10025_with_ijepa_distance.jsonl`
  - Main key: `ijepa_cosine_distance`

- Structured distance:
  `/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/rlinf/assets_isaaclab/all_setting/combined_with_structured_distance/Isaaclab_all_scenarios_10025_with_structured_distance.jsonl`
  - Main key: `structured_distance`
  - Components: cube position, cube layout, cube rpy, camera position, camera rotation, table asset
  - Weight summary is stored at:
    `/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/rlinf/assets_isaaclab/all_setting/combined_with_structured_distance/structured_distance_summary.json`

- 19-stage curriculum variants already generated:
  - `curriculum_19stage_10025_dynamic_top10`
  - `curriculum_19stage_10025_ijepa`
  - `curriculum_19stage_10025_structured`
  - `curriculum_19stage_10025_pixel_l1`
  - `curriculum_19stage_10025_imagebind`
  - `curriculum_19stage_10025_siglip2`

Note: there is no clearly named `fusion` or `fused` distance file in the asset directory. Previous "融合" discussion was usually approximated using `structured_distance`, unless a new fused manifest is explicitly specified.

## 3. Subset5000 Evaluation

Subset file:

`/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/rlinf/assets_isaaclab/all_setting/eval_subsets/Isaaclab_eval_subset5000_cosine_distance_near1000_mid2000_midfar2000_from0.70.jsonl`

Groups:

- `nearest`: 1000 settings
- `middle`: 2000 settings
- `mid_far`: 2000 settings

Grouped success rates:

| Model | nearest | middle | mid_far | total |
|---|---:|---:|---:|---:|
| Base pi05 | `477/1000 = 47.70%` | `224/2000 = 11.20%` | `265/2000 = 13.25%` | `966/5000 = 19.32%` |
| nearest100 step400 | `583/1000 = 58.30%` | `483/2000 = 24.15%` | `649/2000 = 32.45%` | `1715/5000 = 34.30%` |
| mix50 step800 | `585/1000 = 58.50%` | `450/2000 = 22.50%` | `540/2000 = 27.00%` | `1575/5000 = 31.50%` |

## 4. Additional 20260714 Curriculum Model

Training dir:

`/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/logs/20260714-03:48:19-(240step)isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum`

`global_step_520` subset5000 eval:

`/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/logs/all_settings_exact_once_eval/single-ray-resumable-8x4090-rlinf_curriculum_20260714_240step_global_step_520_max450_subset5000_cosine_distance-isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum`

Result:

- Total: `1641 / 5000 = 32.82%`
- nearest: `579 / 1000 = 57.90%`
- middle: `468 / 2000 = 23.40%`
- mid_far: `594 / 2000 = 29.70%`

`global_step_320` subset5000 eval had a partial snapshot when middle reached at least 1000 evaluated settings:

- Snapshot JSON:
  `/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/logs/all_settings_exact_once_eval/single-ray-resumable-8x4090-rlinf_curriculum_20260714_240step_global_step_320_max450_subset5000_cosine_distance-isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum/analysis_subsets/snapshot_middle_1000_success_rates.json`
- Middle first 1000 ids:
  `/inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/logs/all_settings_exact_once_eval/single-ray-resumable-8x4090-rlinf_curriculum_20260714_240step_global_step_320_max450_subset5000_cosine_distance-isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum/analysis_subsets/snapshot_middle_1000_ids.txt`
- Snapshot result:
  - total completed: `801 / 2032 = 39.42%`
  - nearest: `589 / 1000 = 58.90%`
  - middle: `212 / 1032 = 20.54%`
  - mid_far: `0 / 0`

On the same first 1000 middle ids:

- step320: `205 / 1000 = 20.50%`
- step800: `219 / 1000 = 21.90%`

On nearest 1000 ids:

- step320: `589 / 1000 = 58.90%`
- step800: `585 / 1000 = 58.50%`

## 5. Original Cosine Distance Sampling

Nearest small subsets:

| Subset | Base pi05 | nearest100 step400 | mix50 step800 |
|---|---:|---:|---:|
| nearest 60 | `37/60 = 61.67%` | `44/60 = 73.33%` | `49/60 = 81.67%` |
| nearest 100 | `58/100 = 58.00%` | `70/100 = 70.00%` | `78/100 = 78.00%` |

Cosine rank `101-600`:

| Model | Success |
|---|---:|
| step320 from 20260714 model | `310/500 = 62.00%` |
| nearest100 step400 | `305/500 = 61.00%` |
| mix50 step800 | `303/500 = 60.60%` |

Random cosine-ranked sampling summaries:

| Sampling rule | Base pi05 mean | nearest100 step400 mean | mix50 step800 mean |
|---|---:|---:|---:|
| rank > 5000, 10 x 1000 | `12.53%` | `32.25%` | `27.81%` |
| rank > 7000, 10 x 1000 | `12.66%` | `35.42%` | `30.39%` |
| rank > 8000, 10 x 500 | `12.08%` | `37.30%` | `32.72%` |
| rank 1000-9999, 10 x 1000 | `14.42%` | `31.60%` | `27.87%` |
| rank 100-9999, 10 x 1000 | `17.17%` | `34.06%` | `30.02%` |
| rank 100-4999, 10 x 1000 | `22.16%` | `34.79%` | `32.47%` |
| 50/30/20 from 100-999, 1000-4999, 5000-9999 | `30.97%` | `43.78%` | `41.17%` |
| 70/20/10 from 100-999, 1000-4999, 5000-9999 | `37.15%` | `48.52%` | `47.14%` |

## 6. Structured Distance Buckets

Sorted by `structured_distance` from small/easy to large/hard. Each bucket has 1000 settings, except the last one has 25.

| Rank range | Base pi05 | nearest100 step400 | mix50 step800 |
|---|---:|---:|---:|
| 1-1000 | `54.30%` | `76.40%` | `71.60%` |
| 1001-2000 | `51.50%` | `66.10%` | `64.20%` |
| 2001-3000 | `34.00%` | `56.00%` | `52.10%` |
| 3001-4000 | `20.60%` | `53.50%` | `43.70%` |
| 4001-5000 | `9.90%` | `37.20%` | `29.90%` |
| 5001-6000 | `2.70%` | `23.80%` | `19.40%` |
| 6001-7000 | `2.80%` | `14.60%` | `15.40%` |
| 7001-8000 | `0.10%` | `4.60%` | `3.70%` |
| 8001-9000 | `0.00%` | `3.00%` | `3.10%` |
| 9001-10000 | `0.00%` | `2.20%` | `2.10%` |
| 10001-10025 | `0.00%` | `0.00%` | `8.00%` |

Important distinction:

- Structured hardest 1000 means the final 1000 settings after sorting, equivalent to ranks `9026-10025` because there are 10025 total.
- The `9001-10000` bucket is not exactly the same set.

Structured hardest 1000 result:

| Model | Success |
|---|---:|
| Base pi05 | `0 / 1000 = 0.00%` |
| nearest100 step400 | `21 / 1000 = 2.10%` |
| mix50 step800 | `23 / 1000 = 2.30%` |

Structured hard rank `101-1100` when sorted from hard to easy:

| Model | Success |
|---|---:|
| Base pi05 | `0.00%` |
| nearest100 step400 | `27 / 1000 = 2.70%` |
| mix50 step800 | `16 / 1000 = 1.60%` |

Structured middle 1000 when sorted from hard to easy, ranks `4513-5512`:

| Model | Success |
|---|---:|
| Base pi05 | `61 / 1000 = 6.10%` |
| nearest100 step400 | `302 / 1000 = 30.20%` |
| mix50 step800 | `249 / 1000 = 24.90%` |

## 7. I-JEPA / V-JEPA Buckets

Sorted by `ijepa_cosine_distance` from small/easy to large/hard. Each bucket has 1000 settings, except the last one has 25.

| Rank range | Base pi05 | nearest100 step400 | mix50 step800 |
|---|---:|---:|---:|
| 1-1000 | `48.60%` | `58.40%` | `58.40%` |
| 1001-2000 | `22.10%` | `40.60%` | `35.40%` |
| 2001-3000 | `18.40%` | `33.70%` | `30.50%` |
| 3001-4000 | `17.20%` | `31.20%` | `28.10%` |
| 4001-5000 | `14.40%` | `27.00%` | `25.40%` |
| 5001-6000 | `13.10%` | `26.20%` | `24.70%` |
| 6001-7000 | `13.30%` | `30.60%` | `26.50%` |
| 7001-8000 | `10.90%` | `32.00%` | `27.00%` |
| 8001-9000 | `9.80%` | `27.40%` | `24.20%` |
| 9001-10000 | `7.50%` | `29.40%` | `24.30%` |
| 10001-10025 | `24.00%` | `36.00%` | `36.00%` |

I-JEPA / V-JEPA 500-size buckets:

| Rank range | Base pi05 | nearest100 step400 | mix50 step800 |
|---|---:|---:|---:|
| 1-500 | `56.00%` | `63.80%` | `62.80%` |
| 501-1000 | `41.20%` | `53.00%` | `54.00%` |
| 1001-1500 | `24.40%` | `41.40%` | `39.20%` |
| 1501-2000 | `19.80%` | `39.80%` | `31.60%` |
| 2001-2500 | `19.40%` | `36.00%` | `32.00%` |
| 2501-3000 | `17.40%` | `31.40%` | `29.00%` |
| 3001-3500 | `17.60%` | `29.40%` | `28.00%` |
| 3501-4000 | `16.80%` | `33.00%` | `28.20%` |
| 4001-4500 | `13.40%` | `27.40%` | `26.60%` |
| 4501-5000 | `15.40%` | `26.60%` | `24.20%` |
| 5001-5500 | `13.40%` | `24.20%` | `23.20%` |
| 5501-6000 | `12.80%` | `28.20%` | `26.20%` |
| 6001-6500 | `12.60%` | `29.00%` | `25.80%` |
| 6501-7000 | `14.00%` | `32.20%` | `27.20%` |
| 7001-7500 | `10.20%` | `32.60%` | `27.40%` |
| 7501-8000 | `11.60%` | `31.40%` | `26.60%` |
| 8001-8500 | `10.40%` | `29.60%` | `25.40%` |
| 8501-9000 | `9.20%` | `25.20%` | `23.00%` |
| 9001-9500 | `7.80%` | `28.20%` | `23.20%` |
| 9501-10000 | `7.20%` | `30.60%` | `25.40%` |
| 10001-10025 | `24.00%` | `36.00%` | `36.00%` |

## 8. Interpretation So Far

1. **Curriculum-trained models improve strongly over base pi05 on broad setting distributions.**
   - Base all-setting success is `17.55%`.
   - nearest100 step400 reaches `33.66%`.
   - mix50 step800 reaches `30.46%`.

2. **nearest100 step400 is usually stronger than mix50 step800 on broad and hard distributions.**
   - On all 10025: `33.66%` vs `30.46%`.
   - On subset5000: `34.30%` vs `31.50%`.
   - On structured hardest 1000: `2.10%` vs `2.30%`, roughly tied at very low success.
   - On structured middle hard bucket: `30.20%` vs `24.90%`.

3. **Structured distance is more monotonic with empirical task difficulty than I-JEPA/V-JEPA distance.**
   - Structured buckets drop from `54.30% / 76.40% / 71.60%` in the easiest bucket to near zero in the hardest buckets.
   - I-JEPA/V-JEPA buckets are less monotonic. The first 1000 are easier, but middle and later buckets fluctuate.

4. **The hardest structured settings are still largely unsolved.**
   - On structured hardest 1000, all models are near `0-2.3%`.
   - This suggests the curriculum helps coverage, but current training does not solve the far structured tail.

5. **Small nearest/easy sets can look much better than broad generalization.**
   - nearest 60/100 scores are high, especially for mix50 step800.
   - These numbers should not be used alone to claim global generalization.

## 9. Handoff Notes

When continuing this work:

1. Keep evaluation claims tied to exact subset definitions.
   - Avoid saying only "hardest 1000" unless the sorting key is named.
   - Distinguish "last 1000" from fixed buckets like `9001-10000`, because total count is `10025`.

2. For future comparison tables, always record:
   - model checkpoint path
   - eval dir
   - scenario source jsonl
   - sorting key
   - rank range or subset ids
   - success numerator and denominator

3. Prefer `structured_distance` for curriculum-stage difficulty analysis if the goal is monotonic task difficulty.
   - I-JEPA/V-JEPA may still be useful as a visual-difference baseline.
   - ImageBind, SigLIP2, and pixel L1 are available as additional comparison baselines, but they should be validated with the same bucket analysis.

4. If a new "fused" score is introduced, create a clearly named jsonl such as:
   - `combined_with_fused_distance/Isaaclab_all_scenarios_10025_with_fused_distance.jsonl`
   - key: `fused_distance`
   - summary: `fused_distance_summary.json`

5. Keep per-setting eval resumable.
   - Existing eval dirs use `per_setting_results/*.json`, `next_chunk.env`, `EVAL_COMPLETE`, and optional summaries.
   - If a job is killed, count existing JSON files and resume from missing ids rather than rerunning all completed settings.

6. Be careful with machine differences.
   - The coding environment and experiment machine share disk but are not always the same machine.
   - GPU failures, container startup failures, and IsaacLab crashes may only be visible from worker logs on the experiment machine.

## 10. Useful Analysis Pattern

The standard analysis pattern is:

1. Load scenario records from a jsonl.
2. Sort by the chosen distance key.
3. Select a rank range or subset ids.
4. Load each model's `per_setting_results/*.json`.
5. Count success as `success_rate > 0` or `num_success > 0`.
6. Report `num_success / num_settings = percentage`.

For quick checks, avoid relying on only `summary.json`; inspect `per_setting_results` directly when possible.
