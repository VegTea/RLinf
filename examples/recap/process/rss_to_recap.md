# RSS 数据转换为 RECAP 可用格式

这份文档说明如何把 RSS/YAM 任务的 raw LeRobot 数据整理成当前 RECAP 流程可读取的 processed dataset view。转换过程不会复制大文件，也不会修改 raw 数据；脚本只在输出目录下创建 `data/`、`videos/` 的符号链接，并优先复制 raw 数据中已有的真实 `meta/`。只有 raw bucket 缺少完整 `meta/` 时，脚本才会按 parquet 重建最小 `meta/`。

## 适用数据结构

当前脚本默认处理三桶 RSS 数据：

```text
<raw_root>/
├── expert-data/
│   ├── data/
│   ├── meta/
│   └── videos/
├── success-and-hil-data/
│   ├── data/
│   ├── meta/
│   └── videos/
└── failure-data/
    ├── data/
    └── videos/
```

三类数据在 RECAP 中的含义：

- `expert-data`：专家成功数据，按 `type: sft` 处理。
- `success-and-hil-data`：成功或 HITL 修正数据，当前第一版也按 `type: sft` 处理。
- `failure-data`：失败轨迹，要求 parquet 中已有 `reward` 列，按 `type: reward` 处理。

当前 YAM 插电池任务中，`failure-data` 的 reward 约定是中间步 `-1`，最后一步 `-2000`。如果新任务失败惩罚不同，需要同步修改 returns 配置里的 `failure_reward`，并确认 raw reward 列本身是否已经符合预期。

## RECAP 需要的 processed 结构

转换后每个桶会变成一个独立 LeRobot dataset：

```text
data/recap/<task_name>/
├── expert/
│   ├── data -> <raw_root>/expert-data/data
│   ├── videos -> <raw_root>/expert-data/videos
│   └── meta/
├── success_hil/
│   ├── data -> <raw_root>/success-and-hil-data/data
│   ├── videos -> <raw_root>/success-and-hil-data/videos
│   └── meta/
└── failure/
    ├── data -> <raw_root>/failure-data/data
    ├── videos -> <raw_root>/failure-data/videos
    └── meta/
```

每个 `meta/` 至少包含：

- `info.json`
- `tasks.jsonl`
- `episodes.jsonl`

如果 raw dataset 中存在完整 `meta/`，脚本会原样复制该目录。若 raw bucket 缺失 `meta/`，脚本会重建 `info.json`、`tasks.jsonl` 和 `episodes.jsonl`；如果此时 raw dataset 中存在 `episodes_stats.jsonl`，也会复制到 processed view 的 `meta/` 下。

## Step 1: 生成 processed view

以 insert-mouse-battery 为例：

```bash
.venv/bin/python examples/recap/process/prepare_yam_recap_view.py \
  --raw-root /inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/raw/insert-mouse-battery \
  --task-name insert-mouse-battery \
  --output-root data/recap/insert_mouse_battery \
  --force
```

如果不传 `--output-root`，脚本会默认输出到：

```text
data/recap/<task-name-with-underscores>
```

例如 `--task-name insert-mouse-battery` 会输出到 `data/recap/insert_mouse_battery`。

处理新任务时，通常只需要替换 `--raw-root` 和 `--task-name`：

```bash
.venv/bin/python examples/recap/process/prepare_yam_recap_view.py \
  --raw-root /path/to/rss/raw/<new-task-name> \
  --task-name <new-task-name> \
  --force
```

如果新任务的三桶目录名不一样，可以显式指定：

```bash
.venv/bin/python examples/recap/process/prepare_yam_recap_view.py \
  --raw-root /path/to/rss/raw/<new-task-name> \
  --task-name <new-task-name> \
  --expert-dir-name expert-data \
  --success-dir-name success-and-hil-data \
  --failure-dir-name failure-data \
  --expert-output-name expert \
  --success-output-name success_hil \
  --failure-output-name failure \
  --force
```

注意：`--force` 会重建 processed view 下每个 bucket 的 `meta/`。如果这个目录里已经有 `returns_*.parquet` 或 `advantages_*.parquet`，重跑脚本后需要重新计算 returns/advantages。

## Step 2: 检查 processed dataset

生成后先做只读检查：

```bash
.venv/bin/python examples/recap/process/check_recap_readiness.py \
  --dataset data/recap/insert_mouse_battery/expert:sft:yam \
  --dataset data/recap/insert_mouse_battery/success_hil:sft:yam \
  --dataset data/recap/insert_mouse_battery/failure:reward:yam \
  --failure-reward -2000
```

重点确认：

- `data/`、`meta/`、`videos/` 都存在。
- parquet 中有 `episode_index`、`frame_index`、`action`、`observation.state`。
- `info.json` 中有三路相机特征：`cam_high`、`cam_left_wrist`、`cam_right_wrist`。
- YAM 的 action/state 维度符合当前适配，通常为 14。
- `failure` 的 reward 预览能看到最后一步失败惩罚。

## Step 3: 准备 returns 配置

可以复制现有 insert-mouse 配置作为新任务模板：

```bash
cp examples/recap/process/config/compute_returns_insert_mouse_battery_yam.yaml \
  examples/recap/process/config/compute_returns_<task_name>_yam.yaml
```

然后修改：

```yaml
data:
  data_root: data/recap/<task_name_with_underscores>
  train_data_paths:
    - dataset_path: expert
      type: sft
    - dataset_path: success_hil
      type: sft
    - dataset_path: failure
      type: reward

  dataset_type: "sft"
  gamma: 1.0
  failure_reward: -2000.0
  tag: <task_name_with_underscores>
  num_workers: 128
```

字段说明：

- `data_root`：Step 1 生成的 processed root。
- `dataset_path`：相对 `data_root` 的 bucket 名。
- `type: sft`：按成功轨迹生成 reward，中间步 `-1`，最后一步 `0`。
- `type: reward`：直接读取 parquet 中已有 `reward` 列，再反向计算 return。
- `tag`：输出文件名的一部分，最终会生成 `meta/returns_<tag>.parquet`。

## Step 4: 计算 returns

运行：

```bash
bash examples/recap/process/run_compute_returns.sh compute_returns_<task_name>_yam
```

完成后，每个 bucket 应该都有：

```text
data/recap/<task_name>/expert/meta/returns_<tag>.parquet
data/recap/<task_name>/success_hil/meta/returns_<tag>.parquet
data/recap/<task_name>/failure/meta/returns_<tag>.parquet
```

对当前 RSS/YAM 三桶策略，期望结果是：

- `expert`：最后一步 reward 为 `0`。
- `success_hil`：最后一步 reward 为 `0`。
- `failure`：最后一步 reward 保留 raw parquet 中的失败惩罚，例如 `-2000`。

## 可选 Step: 切分 train/eval

如果要从每个 bucket 中留出一部分 episode 做 value model eval，可以在 Step 1 生成 processed view 后运行 `split_lerobot_dataset.py`。脚本会把 split 内 episode 重新编号为连续索引，并把视频文件用 symlink 指向源数据，避免复制大视频。

以 insert-mouse-battery 按 5% 尾部 episode 做 eval 为例：

```bash
.venv/bin/python examples/recap/process/split_lerobot_dataset.py \
  --source-root data/recap/insert_mouse_battery/expert \
  --output-root data/recap/insert_mouse_battery_split/expert \
  --eval-ratio 0.05 \
  --mode tail \
  --force

.venv/bin/python examples/recap/process/split_lerobot_dataset.py \
  --source-root data/recap/insert_mouse_battery/success_hil \
  --output-root data/recap/insert_mouse_battery_split/success_hil \
  --eval-ratio 0.05 \
  --mode tail \
  --force

.venv/bin/python examples/recap/process/split_lerobot_dataset.py \
  --source-root data/recap/insert_mouse_battery/failure \
  --output-root data/recap/insert_mouse_battery_split/failure \
  --eval-ratio 0.05 \
  --mode tail \
  --force
```

输出结构为：

```text
data/recap/<task_name>_split/
├── expert/
│   ├── train/
│   └── eval/
├── success_hil/
│   ├── train/
│   └── eval/
└── failure/
    ├── train/
    └── eval/
```

split 后需要重新计算 returns。可以参考 `compute_returns_insert_mouse_battery_yam_split.yaml`：

```yaml
data:
  data_root: data/recap/<task_name>_split
  train_data_paths:
    - dataset_path: expert/train
      type: sft
    - dataset_path: expert/eval
      type: sft
    - dataset_path: success_hil/train
      type: sft
    - dataset_path: success_hil/eval
      type: sft
    - dataset_path: failure/train
      type: reward
    - dataset_path: failure/eval
      type: reward
```

训练 value model 时，对应配置应使用：

```yaml
data:
  data_root: ${oc.env:REPO_PATH}/data/recap/<task_name>_split
  train_data_paths:
    - dataset_path: expert/train
    - dataset_path: success_hil/train
    - dataset_path: failure/train
  eval_data_paths:
    - dataset_path: expert/eval
    - dataset_path: success_hil/eval
    - dataset_path: failure/eval
```

如果希望随机切分而不是尾部切分，可以使用 `--mode random --seed 42`。

## Step 5: 进入 RECAP 后续流程

returns 生成后，后续流程和普通 RECAP 一样：

1. 训练 value model。
2. 用 value checkpoint 计算 advantages。
3. 从转换后的 OpenPI pi0.5 PyTorch checkpoint 开始做 CFG 训练。

对应配置里需要保持一致的字段：

- `data_root` 指向本任务 processed root。
- `returns_tag` 或相关 tag 字段和 Step 3 的 `tag` 一致。
- `robot_type` 使用 `yam`。
- OpenPI 配置使用 YAM dataconfig，例如 `openpi.config_name: pi05_yam`。
- 如果动作维度、相机键、state/action 字段和当前 YAM 插电池数据不同，需要先更新 YAM dataconfig 和 RECAP value/advantage 的 key mapping。

## 常见问题

**raw 数据会被修改吗？**

不会。脚本只读取 raw 数据，并在 processed root 中创建符号链接和新的 `meta/`。

**failure-data 没有 meta 可以吗？**

可以作为临时 fallback。脚本会用 expert dataset 的 `info.json` 作为参考，为 failure 重建 `meta/info.json`、`tasks.jsonl` 和 `episodes.jsonl`。如果原始 HF 仓库或数据发布源提供了真实 `failure-data/meta`，应先补齐 raw 数据，再重新运行转换脚本，避免使用推断出来的 meta。

**success-and-hil-data 没有 is_success 可以吗？**

可以。当前策略把它作为成功/HITL 数据，使用 `type: sft`，最后一步 reward 设为 `0`。

**只有 expert 和 failure，没有 success/HIL 怎么办？**

当前脚本要求三桶都存在。最小做法是先准备一个空或合并后的 success/HIL bucket；更干净的做法是扩展脚本，让 bucket 列表可选。

**重跑转换脚本后 returns 不见了？**

这是预期行为。`--force` 会重建 `meta/`，因此会删除之前生成的 sidecar 文件。重跑转换后需要重新执行 `compute_returns.py`。
