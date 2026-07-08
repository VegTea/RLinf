# RLinf Checkpoint 转 OpenPI Policy Server 部署

这份文档带你把 RLinf 保存的 OpenPI / RECAP checkpoint 转成 OpenPI websocket policy server 可以加载的目录，并用不改变通信协议的方式部署 RECAP positive guidance 推理。

适用场景：

- 你已有 RLinf 训练保存的 actor checkpoint，例如 `actor/model_state_dict/full_weights.pt`。
- 你想使用 OpenPI 的 websocket + msgpack policy server / client 通信方式。
- 你的模型来自 RECAP / CFG 训练，需要在推理时强制走 `Advantage: positive` 的 condition + uncondition guidance。

## 目录结构

RLinf 保存的 actor checkpoint 通常长这样：

```text
global_step_<N>/
└── actor/
    ├── dcp_checkpoint/
    └── model_state_dict/
        └── full_weights.pt
```

OpenPI policy server 更适合加载这样的目录：

```text
openpi_deploy_ckpt/
├── model.safetensors
└── assets/<task>/<repo_id>/norm_stats.json
```

其中：

- `full_weights.pt` 是 RLinf/FSDP 合并后的 PyTorch 权重。
- `model.safetensors` 是 OpenPI PyTorch policy loader 常用的权重文件。
- `norm_stats.json` 必须和训练数据一致，否则动作归一化会错。

## 第一步：准备路径

在 RLinf 仓库根目录下设置路径。

```bash
export RLINF_ROOT=/path/to/RLinf
cd "$RLINF_ROOT"

export RLINF_ACTOR_CKPT=/path/to/global_step_<N>/actor/model_state_dict/full_weights.pt
export NORM_STATS=/path/to/assets/<task>/<repo_id>/norm_stats.json
export OUT_DIR=/path/to/openpi_deploy_ckpt
export REPO_ID=assets/<task>/<repo_id>
```

YAM tower-of-hanoi 的例子：

```bash
export RLINF_ROOT=/inspire/ssd/project/gjjproject/czxs24230043/RLinf
cd "$RLINF_ROOT"

export RLINF_ACTOR_CKPT=/inspire/qb-ilm/project/gjjproject/public/xhc/phase2/yam_tower-of-hanoi-game_cfg_openpi-20260706-15:36:31/yam_tower-of-hanoi-game_cfg_openpi/checkpoints/best_eval_action_mse/global_step_41000_eval_action_mse_0.000680/actor/model_state_dict/full_weights.pt
export NORM_STATS=/inspire/ssd/project/gjjproject/czxs24230043/RLinf/checkpoints/torch/yam_pi05_tower_of_hanoi_game_199999/assets/tower-of-hanoi-game/expert-success-hil-suffix-mix-data/norm_stats.json
export OUT_DIR=$RLINF_ROOT/checkpoints/openpi_policy_server/yam_tower_of_hanoi_game_step41000
export REPO_ID=assets/tower-of-hanoi-game/expert-success-hil-suffix-mix-data
```

What this does:

1. `RLINF_ACTOR_CKPT` 指向 RLinf 保存的合并权重。
2. `NORM_STATS` 指向训练时使用的 normalization stats。
3. `OUT_DIR` 是最终部署目录。
4. `REPO_ID` 必须和训练配置里的 `actor.model.openpi_data.repo_id` 一致。

## 第二步：转换权重

用 OpenPI 的模型结构重新保存 safetensors。这样可以正确处理 tied/shared weights，比直接 `save_file(state_dict)` 更接近 OpenPI 原生格式。

```bash
mkdir -p "$OUT_DIR/$REPO_ID"

.venv/bin/python - <<'PY'
import os
import pathlib
import shutil

import safetensors.torch
import torch

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from openpi.models_pytorch import pi0_pytorch

actor_ckpt = pathlib.Path(os.environ["RLINF_ACTOR_CKPT"])
norm_stats = pathlib.Path(os.environ["NORM_STATS"])
out_dir = pathlib.Path(os.environ["OUT_DIR"])
repo_id = os.environ["REPO_ID"]

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / repo_id).mkdir(parents=True, exist_ok=True)

train_cfg = get_openpi_config(
    "pi05_yam",
    model_path=str(out_dir),
    data_kwargs={"repo_id": repo_id},
)
model = pi0_pytorch.PI0Pytorch(config=train_cfg.model)

state_dict = torch.load(actor_ckpt, map_location="cpu")
model.load_state_dict(state_dict, strict=True)
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

safetensors.torch.save_model(
    model,
    str(out_dir / "model.safetensors"),
    metadata={"format": "pt"},
)
shutil.copy2(norm_stats, out_dir / repo_id / "norm_stats.json")

print(f"Saved OpenPI checkpoint to: {out_dir}")
PY
```

What this does:

1. 实例化和训练一致的 `pi05_yam` OpenPI PyTorch 模型。
2. 读取 RLinf 的 `full_weights.pt`。
3. 用 `load_state_dict(strict=True)` 检查权重完全匹配。
4. 保存 `model.safetensors`。
5. 把 `norm_stats.json` 放到 OpenPI loader 会读取的位置。

如果你的任务不是 YAM，需要替换 `get_openpi_config("pi05_yam", ...)`、`REPO_ID` 和模型相关参数。转换后，最终目录应至少包含：

```text
$OUT_DIR/
├── model.safetensors
└── $REPO_ID/
    └── norm_stats.json
```

## 第三步：验证转换结果

先确认文件存在。

```bash
ls -lh "$OUT_DIR/model.safetensors"
ls -lh "$OUT_DIR/$REPO_ID/norm_stats.json"
```

再用 OpenPI 模型加载一次。

```bash
.venv/bin/python - <<'PY'
import os
import pathlib

import safetensors.torch
from openpi.models_pytorch import pi0_pytorch

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

out_dir = pathlib.Path(os.environ["OUT_DIR"])
repo_id = os.environ["REPO_ID"]

train_cfg = get_openpi_config(
    "pi05_yam",
    model_path=str(out_dir),
    data_kwargs={"repo_id": repo_id},
)
model = pi0_pytorch.PI0Pytorch(config=train_cfg.model)
missing, unexpected = safetensors.torch.load_model(
    model,
    str(out_dir / "model.safetensors"),
    strict=False,
)

print("missing:", sorted(missing))
print("unexpected:", sorted(unexpected))
PY
```

你应该看到：

```text
missing: []
unexpected: []
```

## 第四步：启动普通 OpenPI Server

如果你的 checkpoint 不是 RECAP / CFG 训练得到的模型，可以使用 OpenPI 原生 policy server 和普通 policy loader。

RECAP / CFG checkpoint 不建议只用普通 `PI0Pytorch` server。普通 server 不会执行 condition + uncondition 两路推理，也不会强制 `Advantage: positive`。

## 第五步：启动 RECAP Positive Guidance Server

使用 RLinf 提供的兼容 server：

```bash
.venv/bin/python toolkits/standalone_eval_scripts/openpi/recap_policy_server.py \
  --checkpoint-dir "$OUT_DIR" \
  --config-name pi05_yam \
  --repo-id "$REPO_ID" \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda:0 \
  --guidance-type positive \
  --guidance-scale 1.0 \
  --positive-only-conditional
```

What this does:

1. 继续使用 OpenPI 的 `WebsocketPolicyServer`。
2. 保持 websocket + msgpack 通信协议不变。
3. 只替换 server 内部 policy 为 RLinf 的 `OpenPi0ForCFGActionPrediction`。
4. 对每个请求自动构造：

   ```text
   <task prompt>
   Advantage: positive
   ```

   和

   ```text
   <task prompt>
   Advantage: negative
   ```

5. 推理时强制使用 positive guidance：

   ```text
   v = (1 - guidance_scale) * v_uncond + guidance_scale * v_positive
   ```

部署后，OpenPI client 仍然按原来的协议发请求。你不需要在 client 侧新增 `positive_guidance_prompt` 或 `negative_guidance_prompt`。

## 请求格式要求

通信协议不变，但 observation 内容必须能被 `pi05_yam` transform 识别。请求里至少需要包含：

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
observation.state
task 或 prompt
```

如果你的 client 使用 OpenPI 原生嵌套格式，保持原格式即可。server 会从 `task` 或 `prompt` 中读取任务描述，并在内部生成 guidance prompt。

## 公网部署注意事项

不要把未加保护的 websocket policy server 直接裸露到公网。至少做一层访问控制：

- 用 VPN / 内网穿透白名单限制访问源。
- 或用 Nginx / Caddy 做 TLS 终止和鉴权。
- 或在云安全组里只放行你的机器人客户端 IP。

OpenPI websocket server 本身只负责模型推理，不提供用户鉴权、限流或 TLS。

## 打包下载

转换完成后，可以打包部署目录。

```bash
tar -cf openpi_deploy_ckpt.tar -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
```

在部署机上解压：

```bash
tar -xf openpi_deploy_ckpt.tar
```

然后把 `--checkpoint-dir` 指向解压后的目录。

## 常见问题

### 为什么没有 `assets/`？

RLinf 的 actor checkpoint 默认只保存训练状态和模型权重，不会自动保存 OpenPI 的 normalization stats。你需要手动把训练时对应的 `norm_stats.json` 放到 `$OUT_DIR/$REPO_ID/norm_stats.json`。

### 为什么没有 `model.safetensors`？

RLinf/FSDP 保存的是 `actor/model_state_dict/full_weights.pt`。OpenPI server 更常见的格式是 `model.safetensors`，因此需要转换。

### 能不能直接改 prompt 成 `Advantage: positive`？

不建议。RECAP / CFG 推理不是单路 prompt 推理，而是同时计算 uncondition 和 positive condition，再按 `guidance_scale` 混合。你需要使用 `recap_policy_server.py` 或等价的 CFG policy implementation。

### 客户端协议要改吗？

不用。`recap_policy_server.py` 继续使用 OpenPI 的 `WebsocketPolicyServer`，所以 websocket/msgpack 协议不变。
