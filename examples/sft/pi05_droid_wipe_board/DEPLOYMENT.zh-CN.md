# π0.5-DROID 擦白板模型部署说明

本文说明如何在一台有 NVIDIA GPU 的工作站上，将本仓库训练的
`openpi_pytorch` π0.5-DROID 擦白板 checkpoint 部署为 OpenPI WebSocket
policy server，并与 `droid-infra` 真机评测端对接。

适用的模型导出目录必须包含：

```text
<CHECKPOINT_DIR>/
├── model.safetensors
├── config.json
└── assets/wipe_board_v1_zed196_force/norm_stats.json
```

左、右相机模型是两个不同的 checkpoint：部署时必须让
`EXTERIOR_CAMERA` 与训练时使用的外部相机一致。

## 1. 环境与 tokenizer

在部署工作站准备 RLinf 环境，并安装与训练时兼容的 OpenPI PyTorch
依赖。启动脚本使用当前仓库的 `.venv/bin/python`；如使用其他虚拟环境，
通过 `PYTHON_BIN` 显式指定解释器。

PaliGemma tokenizer 不在 inference checkpoint 中。将它下载到一个当前用户
可写的本地缓存目录；以下命令关闭代理：

```bash
TOKENIZER_CACHE=/home/user/Workspace/openpi_cache
mkdir -p "$TOKENIZER_CACHE/big_vision"

curl --noproxy '*' -L \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
  -o "$TOKENIZER_CACHE/big_vision/paligemma_tokenizer.model"

sha256sum "$TOKENIZER_CACHE/big_vision/paligemma_tokenizer.model"
```

文件 SHA256 应为：

```text
8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6
```

文件结构必须为：

```text
/home/user/Workspace/openpi_cache/
└── big_vision/
    └── paligemma_tokenizer.model
```

后续 `OPENPI_DATA_HOME` 必须指向 `openpi_cache` 根目录，而不是 tokenizer
文件本身或 `big_vision` 子目录。

## 2. 启动 policy server

进入 RLinf 仓库并选择一个可用 GPU。以下使用端口 `8080`，并假设 checkpoint
已经解压到 `/path/to/pi05_droid_right_h200_step18000`：

```bash
cd /home/user/Workspace/RLinf

OPENPI_DATA_HOME=/home/user/Workspace/openpi_cache \
CHECKPOINT_DIR=/path/to/pi05_droid_right_h200_step18000 \
EXTERIOR_CAMERA=right \
CUDA_VISIBLE_DEVICES=0 \
SERVER_PORT=8080 \
bash examples/sft/pi05_droid_wipe_board/run_rlinf_server.sh
```

部署左相机模型时，只替换模型路径和相机选择：

```bash
OPENPI_DATA_HOME=/home/user/Workspace/openpi_cache \
CHECKPOINT_DIR=/path/to/pi05_droid_left_h200_step18000 \
EXTERIOR_CAMERA=left \
CUDA_VISIBLE_DEVICES=0 \
SERVER_PORT=8080 \
bash examples/sft/pi05_droid_wipe_board/run_rlinf_server.sh
```

服务会在加载权重、norm stats 和 tokenizer 后才监听 WebSocket。可用以下命令
检查 TCP/HTTP 健康端点：

```bash
curl http://127.0.0.1:8080/healthz
```

期望返回：

```text
OK
```

服务的首个请求通常明显更慢。默认设置
`TORCH_COMPILE_DISABLE=1`，避免 4090 上的长时间首请求编译；在已充分 warm-up
的固定部署环境中，才考虑设置 `TORCH_COMPILE_DISABLE=0`。

## 3. 输入、输出与控制语义

服务端接收原始 RGB 图像和当前机器人状态，再在服务端执行 OpenPI transform：

| 项目 | 形状/语义 |
| --- | --- |
| 选中的外部相机 | `uint8 H×W×3`，left 使用 `observation/exterior_image_1_left`；right 使用 `observation/exterior_image_2_left` |
| 腕部相机 | `uint8 H×W×3`，键名 `observation/wrist_image_left` |
| 关节状态 | `observation/joint_position`，7 个 Franka 当前关节绝对位置，单位 rad |
| 夹爪状态 | `observation/gripper_position`，形状 `(1,)` |
| 语言 | `prompt`，训练集使用 `wipe the whiteboard` |

模型最终使用三路 `224×224` 图像：

```text
[base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb]
= [选中外部相机, 腕部相机, 全零图像]
```

第三路图像的 mask 为 `false`，不是第二个外部相机。当前模型不使用
cartesian pose、force/torque、action history 或未选中的外部相机；droid-infra
即使发送这些字段，服务端也不会输入模型。

模型内部预测 `(15, 8)` action chunk：前 7 维为 15 Hz joint velocity，第 8 维为
absolute gripper position。WebSocket wrapper 使用本次请求的当前关节位置积分
前 7 维，因此返回给客户端的是 `(15, 8)` absolute joint position：

```text
q_target[t] = q_current + cumsum(v / 15 Hz)[t]
gripper_target[t] = model_gripper[t]
```

metadata 会声明：

```text
action_space: joint_position
native_model_action_space: joint_velocity
control_frequency_hz: 15
action_horizon: 15
```

所以 droid-infra 必须以 `joint_position` action space 控制机器人，不能把服务端
返回值再次当作 velocity 积分或乘以控制频率。

## 4. droid-infra 对接

在 droid-infra 的 OpenPI evaluation client 中，设置：

```text
policy endpoint: ws://<GPU_WORKSTATION_IP>:8080
instruction: wipe the whiteboard
control_hz: 15
action space: joint_position
gripper action space: position
```

客户端外部相机 ID 必须对应训练相机：

| 模型 | `EXTERIOR_CAMERA` | droid-infra 应传入第一外部相机槽的实际相机 |
| --- | --- | --- |
| right checkpoint | `right` | 真机右侧外部相机 |
| left checkpoint | `left` | 真机左侧外部相机 |

启动前先确认相机画面方向；left/right 的含义来自数据采集的
`exterior_image_1_left` 与 `exterior_image_2_left` 键，不应仅按物理安装位置
猜测。

### Execute horizon

`exec_horizon` 不会改变模型的 15-step chunk，而是指定每次向 server 请求 chunk
后实际连续执行前多少步：

| `exec_horizon` | 15 Hz 下开环执行时间 | 建议用途 |
| ---: | ---: | --- |
| 1 | 0.067 秒 | 首次真机验证；闭环性最强 |
| 3 | 0.20 秒 | 完成动作方向和安全验证后的首个折中值 |
| 5 | 0.33 秒 | 推理时间无法满足 15 Hz 时的折中 |
| 15 | 1.0 秒 | 不建议用于首次接触式擦白板测试 |

每个 chunk 的未执行后段会被丢弃；下一次请求使用新相机帧和新的当前关节位置。
擦白板有持续接触与位置误差，建议从 `--exec-horizon 1` 开始。

以 droid-infra 的专用 client 为例：

```bash
cd /path/to/droid-infra

PYTHONPATH=. python3 scripts/server/run_openpi_wipeboard_policy_client.py \
  --server-host <GPU_WORKSTATION_IP> \
  --server-port 8080 \
  --instruction "wipe the whiteboard" \
  --action-space joint_position \
  --exec-horizon 1 \
  --max-steps 300 \
  --execute
```

先去掉 `--execute` 做一次 dry run，确认相机、WebSocket 连接、action shape 和输出
范围正常，再执行真机动作。真机首测还建议设置 droid-infra 的
`--position-blend` 为小于 `1.0` 的保守值，并逐步提高到 `1.0`。

## 5. 记录部署 observation

为排查真机输入是否与训练一致，可在 server 端开启同步记录：

```bash
cd /home/user/Workspace/RLinf

OPENPI_DATA_HOME=/home/user/Workspace/openpi_cache \
CHECKPOINT_DIR=/path/to/pi05_droid_right_h200_step18000 \
EXTERIOR_CAMERA=right \
OBSERVATION_RECORD_DIR="$PWD/outputs/pi05_droid_observation_debug" \
OBSERVATION_RECORD_FPS=15 \
SERVER_PORT=8080 \
bash examples/sft/pi05_droid_wipe_board/run_rlinf_server.sh
```

每次启动会创建：

```text
outputs/pi05_droid_observation_debug/
└── session_<UTC时间>_pid<PID>/
    ├── metadata.json
    ├── observations.jsonl
    ├── exterior.mp4
    ├── wrist.mp4
    └── model_inputs_224.mp4
```

- `observations.jsonl`：每个 inference request 一行；包含 UTC、Unix、monotonic
  时间戳，非图像 observation 和图像对应的 video frame index。
- `exterior.mp4`：模型选中的原始外部相机画面。
- `wrist.mp4`：原始腕部相机画面。
- `model_inputs_224.mp4`：最终送进模型的三联画
  `[exterior, wrist, masked zero]`。

MP4 使用 H.264/yuv420p，可直接在 VS Code 中打开。请使用 `Ctrl+C` 正常停止 server，
以完成 MP4 索引写入。

记录单位是 inference request，不是 droid-infra 的每个控制步：当
`exec_horizon=1` 时两者相同；当 `exec_horizon>1` 时，中间开环执行步不会发到
server，因此也不会被记录。

## 6. 常见问题

### `norm_stats.json not found`

新导出包应含有：

```text
<CHECKPOINT_DIR>/assets/wipe_board_v1_zed196_force/norm_stats.json
```

若使用旧包，显式设置：

```bash
NORM_STATS_DIR=/path/to/wipe_board_v1_zed196_force
```

### `PermissionError: /inspire` 或 tokenizer 下载失败

部署工作站不能使用训练服务器的 `/inspire/...` 路径。按第 1 节下载 tokenizer，
并显式设置可写的 `OPENPI_DATA_HOME`。

### 输出的关节动作形状不对

本服务要求并返回 `(15, 8)`。若 droid-infra 报错，先确认：

```text
joint_position: (7,)
gripper_position: (1,)
actions: (15, 8)
```

### 机器人的动作方向不对或抖动

停止执行并先检查：

1. 是否将 left/right checkpoint 与真实相机对应错。
2. 是否向服务传入当前、单位为 rad 的 7D joint position。
3. droid-infra 是否使用 `joint_position`，而非 `joint_velocity`。
4. `exec_horizon` 是否从 1 开始，且是否使用了保守的 `position_blend`。
5. `model_inputs_224.mp4` 是否与预期相机视角和裁剪结果一致。
