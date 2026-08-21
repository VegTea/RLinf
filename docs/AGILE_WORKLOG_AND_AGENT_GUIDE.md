# AGILE 工作记录与后续 Agent 指南

这份文档用于记录目前围绕 IsaacLab Franka stack cube 任务做过的改进、实验设计、数据文件、训练/评测/可视化命令，以及后续让 agent 接手时需要注意的事项。

后续和 agent 对话时，可以直接让它先读这个文件：

```text
docs/AGILE_WORKLOG_AND_AGENT_GUIDE.md
```

## 1. 当前研究故事

建议方法名：

```text
AGILE: Adaptive Generalization via Iterative Learning and Expansion for VLA Models
```

中文表述：

```text
AGILE：面向 VLA 模型的迭代扩展式自适应泛化方法
```

核心故事不是“大而泛的课程学习”或“持续学习”，而是更窄的泛化问题：

- 我们希望 VLA policy 在更丰富的任务配置上泛化。
- 任务配置包括 cube 初始位姿、table visual asset、相机视角等 setting。
- world model / embedding 用于描述不同 setting 之间的变化和难易关系。
- 训练不是一次性把所有 setting 打开，也不是完全人工筛选阶段。
- 系统会根据当前 policy 的成功率自动切换阶段，并逐步扩展下一阶段训练分布。
- 仿真实验中主要是固定 stage 内做 RL，达到成功率阈值或 max stage step 后进入下一阶段。
- 真机实验中还有主动补数据闭环：用 DAgger 把 rollout 中表现差或执行质量不好的数据加入纠正，再回到训练集。
- 日志中会记录新加入区域和累计区域的成功率，从而分析哪些区域仍然薄弱；在真机设置下，这些薄弱区域可以转化为下一轮 DAgger 纠正数据。

一句话 TL;DR：

```text
AGILE uses world-model-derived task embeddings and policy success feedback to automatically expand the next training distribution needed for stronger VLA generalization.
```

中文一句话：

```text
AGILE 利用 world model 产生的任务 embedding 和 policy 成功率反馈，自动扩展下一阶段最需要的训练分布，从而提升 VLA 模型泛化能力。
```

更准确的实验叙述：

- Simulation: stage 内用 RL 训练，成功率达标或达到 `max_steps_per_stage` 后自动扩展。
- Real world: 使用 DAgger 风格的数据聚合，对 rollout 中失败或质量差的片段加入专家纠正，从而主动补充薄弱区域数据。

## 2. 总体实现路线

现在的实现是原 RLinf/IsaacLab pipeline 上的旁路，不应该覆盖原管线。

基本流程：

1. 生成完整 scenario JSONL。每一行是一组可复现的任务 setting。
2. 生成 stage manifest。manifest 显式记录每个阶段新增哪些 id、累计允许哪些 id。
3. 训练 YAML 指向这个 scenario JSONL 和 stage manifest。
4. `ScenarioScheduler` 根据当前 stage 设置 allowed scenario ids。
5. `EnvWorker` 收到 runner 的 stage 更新后，只更新采样范围，不重建 IsaacLab env。
6. 下一次 reset 时从当前 allowed ids 里采样 setting。
7. `EmbodiedRunner` 根据训练成功率决定是否进入下一阶段。

这个方案保留了之前优化过的快速场景替换逻辑。训练过程中不是每个 scenario 都重新启动 IsaacLab，而是在已经启动的 env 内做 reset 和 asset/pose/camera 设置。

## 3. 关键代码改动

### 3.1 ScenarioScheduler

关键文件：

```text
rlinf/envs/isaaclab/scenario_scheduler.py
```

当前支持的 scenario reset mode：

- `sequential`：顺序遍历 active ids。
- `random`：从 active ids 中随机采样。
- `by_id`：只用 YAML 里给定的 `fixed_ids`。
- `external`：从两个外部 id group 按比例采样。

课程/泛化扩展支持两条路径：

- 旧路径：只配置 `thresholds` 时，按 `distance <= current_threshold` 得到 allowed ids。
- 新旁路：配置 `stage_manifest_file` 时，直接读取 manifest 中每个 stage 的 `cumulative_ids`，精确控制每阶段 setting 数量。

manifest 路径的好处：

- 不受浮点阈值边界影响。
- 第一个阶段可以精确是 10 个 setting。
- 后续阶段数量可精确控制，比如 10、20、40、...、200。
- 后续要换阶段划分，只需要重新生成 manifest 和 JSONL，训练代码不需要大改。

`ScenarioScheduler.get_curriculum_state()` 当前会返回：

```text
enabled
stage_index
threshold
allowed_scenarios
total_scenarios
distance_key
stage_source
stage_step
old_scenarios
new_scenarios
mix_old_ratio
mix_new_ratio
is_all_scenarios
```

### 3.2 新旧数据混合采样

`scenario_scheduler.py` 还支持：

```yaml
curriculum:
  sampling:
    mix_schedule:
      - steps: ...
        old_ratio: ...
        new_ratio: ...
```

含义：

- stage 刚切换后，可以让采样更偏向新加入的 incremental ids。
- 例如 `mix50` 类配置会让 old/new 各占一定比例，避免新数据被大量 old cumulative 数据淹没。
- `stage_step` 由 runner 同步给 env worker，再传给 scheduler，用于判断当前处在 mix schedule 的哪个阶段。

### 3.3 EmbodiedRunner 阶段推进

关键文件：

```text
rlinf/runners/embodied_runner.py
```

基础推进逻辑：

- `success_threshold`：达到该成功率视为一次命中。
- `stable_steps`：需要命中多少次后进入下一阶段。
- `require_consecutive_success: false`：命中不要求连续。

例如：

```yaml
success_threshold: 0.75
stable_steps: 3
require_consecutive_success: false
```

表示当前 stage 中累计出现 3 次成功率大于等于 75%，即可进入下一 stage。

后续又加入了 grouped moving average 推进逻辑：

```yaml
promotion:
  mode: grouped_moving_average
  window: 5
  min_steps_per_stage: 20
  max_steps_per_stage: 300
  cumulative_success_thresholds: [...]
  new_success_thresholds: [...]
```

这个模式会分别看：

- 当前累计 allowed set 的成功率；
- old cumulative ids 的成功率；
- new incremental ids 的成功率。

这样能判断“新加进来的 setting 是不是已经学会”，而不是只看总成功率。

常见 metrics：

```text
curriculum/stage_index
curriculum/threshold
curriculum/allowed_scenarios
curriculum/stage_steps
curriculum/success_hits
curriculum/success_threshold
curriculum/cumulative_success
curriculum/new_increment_success
curriculum/old_cumulative_success
curriculum/cumulative_success_window_mean
curriculum/new_success_window_mean
curriculum/old_success_window_mean
curriculum/can_advance
curriculum/advanced
curriculum/mix_old_ratio
curriculum/mix_new_ratio
```

### 3.4 EnvWorker 诊断日志

关键文件：

```text
rlinf/workers/env/env_worker.py
```

新增/使用的功能：

- `set_scenario_curriculum_stage(stage_index)`：runner 通知 env worker 切阶段。
- `set_scenario_curriculum_progress(stage_index, stage_step)`：同步当前 stage 内训练步数，供 mix schedule 使用。
- `_write_curriculum_scenario_stats(...)`：可选写每个 scenario 的 episode/success 统计。

打开 per-scenario 诊断：

```yaml
env:
  train:
    init_params:
      scenario_reset:
        curriculum:
          diagnostics:
            write_scenario_stats: true
```

输出路径：

```text
<log_dir>/worker_logs/curriculum_scenario_stats_rank_<rank>.jsonl
```

## 4. 数据文件与阶段文件

原始 10025 setting 文件：

```text
rlinf/assets_isaaclab/all_setting/combined_with_distance/Isaaclab_all_scenarios_10025_with_distance.jsonl
```

结构化距离版本：

```text
rlinf/assets_isaaclab/all_setting/combined_with_structured_distance/Isaaclab_all_scenarios_10025_with_structured_distance.jsonl
```

10阶段/200 setting 静态版本：

```text
rlinf/assets_isaaclab/all_setting/curriculum_10stage_200/
```

里面的重要文件：

```text
Isaaclab_10stage_200_with_distance.jsonl
Isaaclab_10stage_200_stage_manifest.json
stage_sets/incremental/stage_00.jsonl ... stage_09.jsonl
stage_sets/cumulative/stage_00.jsonl ... stage_09.jsonl
```

10阶段/200 setting 的累计数量：

```text
10,20,40,60,80,100,120,140,170,200
```

静态版本的阈值：

```text
0.00989311933517456
0.012294411659240723
0.017917275428771973
0.020775020122528076
0.024299979209899902
0.026171326637268066
0.030017435550689697
0.03304260969161987
0.03589385747909546
0.040190041065216064
```

10阶段/200 setting dynamic top10 版本：

```text
rlinf/assets_isaaclab/all_setting/curriculum_10stage_200_dynamic_top10/
```

19阶段/10025 setting dynamic top10 版本：

```text
rlinf/assets_isaaclab/all_setting/curriculum_19stage_10025_dynamic_top10/
```

19阶段/10025 setting structured 版本：

```text
rlinf/assets_isaaclab/all_setting/curriculum_19stage_10025_structured/
```

## 5. 数据生成脚本

### 5.1 固定 distance 排序生成 manifest

脚本：

```text
examples/embodiment/generate_curriculum_stage_manifest.py
```

用途：

- 从带 distance 的 JSONL 中按 `distance_key` 从小到大排序；
- 选择前 N 个 setting；
- 按给定 counts 生成 stage manifest；
- 同时写 incremental/cumulative stage JSONL。

示例：重新生成 10阶段/200 setting 静态版本：

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf

.venv/bin/python examples/embodiment/generate_curriculum_stage_manifest.py \
  --source rlinf/assets_isaaclab/all_setting/combined_with_distance/Isaaclab_all_scenarios_10025_with_distance.jsonl \
  --output-dir rlinf/assets_isaaclab/all_setting/curriculum_10stage_200 \
  --distance-key cosine_distance \
  --counts 10,20,40,60,80,100,120,140,170,200 \
  --prefix Isaaclab_10stage_200
```

### 5.2 dynamic top-k region growing

脚本：

```text
examples/embodiment/generate_dynamic_curriculum_stage_manifest.py
```

用途：

- stage0 从 reference id 附近选择；
- 后续每个 stage 从未选 setting 中选择与已掌握区域最近的 top-k mean distance；
- 这相当于从一个 reference setting 出发逐步扩展任务区域。

### 5.3 base embedding dynamic top-k

脚本：

```text
examples/embodiment/generate_dynamic_curriculum_from_base_embeddings.py
```

用途：

- 给定一组 base embeddings，作为初始已掌握区域；
- 每阶段从候选池中选择最接近 mastered set 的新 setting；
- 每个 setting 会记录：
  - `selected_stage`
  - `base_dynamic_top{k}_mean_cosine_distance`
  - `nearest_mastered_refs`
  - `nearest_mastered_distances`

### 5.4 structured distance

脚本：

```text
examples/embodiment/generate_structured_scenario_distance.py
```

用途：

- 不只依赖 embedding cosine distance；
- 可以用结构化任务因素设计难度，比如 cube pose、camera、table variation 等。

## 6. 训练配置和启动命令

原始非旁路配置：

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml
```

顺序/随机 scenario JSONL 配置：

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_scenarios.yaml
```

旧 19 阶段 threshold 配置：

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum.yaml
```

75% 成功率累计 3 次配置：

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_075threshold_3hits.yaml
```

当前 10阶段/200 setting 相关配置：

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200.yaml
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200_max300.yaml
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200_mix50.yaml
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200_mix50_max300.yaml
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200_dynamic_top10.yaml
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200_dynamic_top10_max300.yaml
```

当前 19 阶段 dynamic 配置：

```text
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_19stage_dynamic_top10.yaml
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_19stage_dynamic_top10_max300.yaml
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_19stage_dynamic_top10_mix50_max300.yaml
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_19stage_dynamic_top10_mix50_max300_32gpu.yaml
```

10阶段/200 setting 静态训练：

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf
bash examples/embodiment/run_embodiment_curriculum_10stage_200.sh
```

10阶段/200 setting dynamic top10 训练：

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf
bash examples/embodiment/run_embodiment_curriculum_10stage_200_dynamic_top10.sh
```

19阶段 dynamic top10 + mix50 + max300 训练：

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf
bash examples/embodiment/run_embodiment_curriculum_19stage_dynamic_top10_mix50_max300.sh
```

32 GPU / 多节点旁路：

```text
examples/embodiment/run_embodiment_ray_multinode_32gpu.sh
examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_19stage_dynamic_top10_mix50_max300_32gpu.yaml
```

注意 batch 和 env 的区别：

- `env.train.total_num_envs: 128` 是并行 IsaacLab env 数。
- `actor.micro_batch_size` 是单次训练 micro batch。
- `actor.global_batch_size` 是 learner 全局 batch。
- 当前常用 VLA 配置一般是：

```yaml
actor:
  micro_batch_size: 32
  global_batch_size: 256
```

## 7. 评测

旧 range eval：

```text
examples/embodiment/eval_curriculum_ranges.sh
```

用途：

- 对每个 stage 做评测；
- 支持 cumulative 和 incremental 两种范围；
- 生成每个范围单独的 scenario JSONL；
- 保存 `summary.csv` 和 `summary.json`。

structured range eval：

```text
examples/embodiment/eval_structured_curriculum_ranges.sh
```

示例：

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf

bash examples/embodiment/eval_structured_curriculum_ranges.sh \
  --mode both \
  --total-num-envs 32 \
  --eval-rollout-epoch 20
```

断点继续：

```bash
bash examples/embodiment/eval_structured_curriculum_ranges.sh \
  --output-dir <已有评测输出目录> \
  --skip-existing
```

输出结构：

```text
<eval_output_dir>/runs.json
<eval_output_dir>/summary.json
<eval_output_dir>/summary.csv
<eval_output_dir>/<mode>/stage_XX/eval_metrics.pt
<eval_output_dir>/<mode>/stage_XX/eval_embodiment.log
<eval_output_dir>/scenario_sets/<mode>/stage_XX.jsonl
```

checkpoint 注意：

- `runner.resume_dir` 用于恢复完整训练 checkpoint。
- `runner.ckpt_path` 用于加载 `.pt` 模型权重，主要用于 eval。
- eval wrapper 默认把 `runner.resume_dir=null`、`runner.ckpt_path=null`，如需指定权重应显式覆盖。

## 8. 可视化

基础渲染脚本：

```text
examples/embodiment/render_stack_cube_all_scenarios.sh
examples/embodiment/render_stack_cube_all_scenarios.py
```

按 stage 抽样渲染脚本：

```text
examples/embodiment/render_stack_cube_incremental_samples.sh
```

渲染旧 19 个 incremental stage，每个 stage 默认 20 个：

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf
bash examples/embodiment/render_stack_cube_incremental_samples.sh
```

渲染新 10阶段/200 setting 的 incremental stage：

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf

bash examples/embodiment/render_stack_cube_incremental_samples.sh \
  --input-dir /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/rlinf/assets_isaaclab/all_setting/curriculum_10stage_200/stage_sets/incremental \
  --output-root /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/logs/curriculum_10stage_200_incremental_renders \
  --samples-per-stage 20
```

说明：

- 这个脚本不自动把任务分配到 8 张卡。
- 它会逐 stage 调用 `render_stack_cube_all_scenarios.py`。
- 如果想指定某张卡：

```bash
CUDA_VISIBLE_DEVICES=3 bash examples/embodiment/render_stack_cube_incremental_samples.sh ...
```

- 如果某个 stage 的 incremental setting 少于 20 个，会全部渲染。
- `--save-wrist` 会额外保存腕部相机图像。
- `--prepare-only` 只生成抽样 JSONL 和 summary，不启动 IsaacLab 渲染。

## 9. 日志怎么看

主日志：

```text
<log_dir>/run_embodiment.log
```

worker 日志：

```text
<log_dir>/worker_logs/EnvGroup/rank_<n>.log
<log_dir>/worker_logs/ActorGroup/rank_<n>.log
<log_dir>/worker_logs/RolloutGroup/rank_<n>.log
```

运行时完整配置快照：

```text
<log_dir>/tensorboard/all/config.yaml
```

常用搜索：

```bash
rg -n "Initialized scenario curriculum|Updated scenario curriculum|Advanced scenario curriculum|curriculum/stage_index|ERROR_DEVICE_LOST|GPU crash|exitcode=-11|Traceback|Exception" <log_dir>
```

判断是否进入下一阶段：

- 看 `run_embodiment.log` 里的：

```text
Initialized scenario curriculum
Updated scenario curriculum stage to ...
Advanced scenario curriculum to stage ...
```

- 或看 TensorBoard：

```text
curriculum/stage_index
curriculum/allowed_scenarios
curriculum/advanced
curriculum/can_advance
```

判断为什么没进入下一阶段：

- 看是否满足 `success_threshold` / `stable_steps`。
- 如果是 grouped moving average，重点看：

```text
curriculum/new_increment_success
curriculum/new_success_window_mean
curriculum/cumulative_success
curriculum/cumulative_success_window_mean
curriculum/cumulative_floor
curriculum/min_steps_per_stage
curriculum/max_steps_per_stage
```

## 10. 常见错误判断

### 10.1 GPU/IsaacLab renderer 崩溃

典型日志：

```text
VkResult: ERROR_DEVICE_LOST
GPU crash is detected
A GPU crash occurred
RuntimeError: IsaacLab subprocess exited while waiting for reset; exitcode=-11
```

这通常不是 JSONL 或 manifest 的 Python 逻辑问题，而是 IsaacLab/Isaac Sim/Vulkan/GPU 侧崩溃，可能原因包括：

- 某张 GPU 状态不稳定；
- Vulkan device lost；
- OOM-like device loss；
- Isaac Sim renderer 多进程初始化不稳定；
- 分布式容器环境和交互式 docker 环境不完全一致。

排查方式：

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/embodiment/check_isaaclab_gpu_smoke.sh \
  --gpus 1 \
  --config isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200
```

如果某张卡反复失败，可以先排除：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 bash examples/embodiment/run_embodiment_curriculum_10stage_200.sh
```

### 10.2 scenario / manifest 配置错误

典型日志：

```text
Curriculum stage manifest has no stages
contains ids not present in scenario_file
scenario_reset.curriculum requires scenario records with distances
Unable to resolve table asset
```

这类问题通常是：

- YAML 指向了错误 JSONL；
- manifest 里的 id 不在 scenario JSONL 中；
- distance key 不存在；
- table asset 路径或 USD/PNG 资源不完整。

## 11. GPU Smoke Test

脚本：

```text
examples/embodiment/check_isaaclab_gpu_smoke.sh
```

用途：

- 模拟训练启动；
- 设置场景；
- 加载模型；
- 跑 action；
- 检查 GPU/IsaacLab/model 交互是否正常。

1 卡测试：

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf

CUDA_VISIBLE_DEVICES=0 bash examples/embodiment/check_isaaclab_gpu_smoke.sh \
  --gpus 1 \
  --config isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200
```

8 卡测试：

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf

bash examples/embodiment/check_isaaclab_gpu_smoke.sh \
  --gpus 8 \
  --config isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200
```

## 12. USD / PNG / table asset 路径

路径解析代码：

```text
rlinf/envs/isaaclab/scenario_loader.py
```

默认 table asset root：

```text
rlinf/assets_isaaclab/SeattleLabTable/color_tables
```

解析逻辑：

- 如果 `table_asset` 是绝对路径，直接用。
- 如果是相对路径，就拼到默认 table asset root 下。
- 如果后缀是 `.usd`，也会尝试 `.usda`。
- 如果后缀是 `.usda`，也会尝试 `.usd`。
- 如果没有后缀，会尝试 `.usd` 和 `.usda`。

注意：

- USD 内部的 PNG/texture 引用可能是相对路径，也可能是绝对路径。
- 如果移动到别的机器后 PNG 找不到，要检查 USD 文件内部引用。
- 如果 USD 内是绝对路径，必须保持同样挂载路径，或者批量改 USD 内部路径。

## 13. 当前重要假设

- 当前主任务是 IsaacLab Franka stack cube。
- `ROBOT_PLATFORM` 通常应是：

```bash
ROBOT_PLATFORM=LIBERO
```

- 当前 actor 输入图像数量是 2：

```yaml
actor:
  model:
    openpi:
      num_images_in_input: 2
```

这通常表示第三视角和腕部视角都进入模型输入。是否保存腕部图像/视频取决于渲染或 video config。

- `env.train.total_num_envs=128` 表示并行 env 数，不是 scenario setting 总数。
- stage0 如果只有 10 个 setting，而并行 env 是 128，那么不同 env 之间必然会重复 setting。
- 当前 scheduler 的 random 采样会按 seed/batch index 采样；不能保证 128 个 env 全唯一。
- `resume_dir` 会恢复模型/优化器等训练 checkpoint，但 curriculum stage/hit 计数是否完全恢复需要看具体实现和日志确认。

## 14. 后续 Agent 工作准则

后续 agent 接手时，优先做这些：

1. 先读这个文档。
2. 执行：

```bash
git status --short
```

3. 不要回退用户或前序 agent 的未提交修改。
4. 先看用户给的具体 config 名，不要默认用最新 YAML。
5. 对训练问题，优先看：

```text
<log_dir>/run_embodiment.log
<log_dir>/tensorboard/all/config.yaml
<log_dir>/worker_logs/
```

6. 对 stage 问题，优先看 manifest 和 stage JSONL，不要只看 YAML thresholds。
7. 新实验尽量新建 YAML 和 wrapper script，不覆盖旧配置。
8. 新的数据划分必须能复现，优先写/改生成脚本，不要手工编辑大 JSONL。
9. 如果出现 `ERROR_DEVICE_LOST` / `exitcode=-11`，先按 GPU/IsaacLab 崩溃排查，不要先怀疑 manifest。
10. 如果要改 story/论文命名，当前推荐围绕 `AGILE` 和 `adaptive generalization expansion`，避免过度使用 broad curriculum / continual learning。

## 15. 常用命令索引

训练 10阶段/200 setting 静态版本：

```bash
bash examples/embodiment/run_embodiment_curriculum_10stage_200.sh
```

训练 10阶段/200 setting dynamic top10：

```bash
bash examples/embodiment/run_embodiment_curriculum_10stage_200_dynamic_top10.sh
```

训练 19阶段 dynamic top10 mix50 max300：

```bash
bash examples/embodiment/run_embodiment_curriculum_19stage_dynamic_top10_mix50_max300.sh
```

渲染旧 19 阶段增量：

```bash
bash examples/embodiment/render_stack_cube_incremental_samples.sh
```

渲染新 10阶段/200 setting 增量：

```bash
bash examples/embodiment/render_stack_cube_incremental_samples.sh \
  --input-dir /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/rlinf/assets_isaaclab/all_setting/curriculum_10stage_200/stage_sets/incremental \
  --output-root /inspire/hdd/global_user/gongjingjing-25039/syfei/corriculum_rl/RLinf/logs/curriculum_10stage_200_incremental_renders \
  --samples-per-stage 20
```

structured range eval：

```bash
bash examples/embodiment/eval_structured_curriculum_ranges.sh --mode both
```

1 卡 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/embodiment/check_isaaclab_gpu_smoke.sh \
  --gpus 1 \
  --config isaaclab_franka_stack_cube_ppo_openpi_pi05_table_curriculum_10stage_200
```

日志快速搜索：

```bash
rg -n "Advanced scenario curriculum|Initialized scenario curriculum|Updated scenario curriculum|ERROR_DEVICE_LOST|GPU crash|exitcode=-11|Traceback|Exception" <log_dir>
```
