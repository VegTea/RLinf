# qzcli 提交 H100 训练任务时 GPU 类型枚举不匹配

记录日期：2026-08-23

## 问题概述

使用 qzcli/MCP 向启智平台提交 GPU 分布式训练任务时，资源查询能够正常找到目标 H100 计算组和 4 卡规格，但创建任务接口在调度前拒绝请求。

```text
InvalidParameter: framework_config[0]: gpu_type "H100" does not match
logic_compute_group "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7"
real gpu type "NVIDIA_H100_SXM_80G"
```

任务没有被创建。因此该问题发生在创建参数校验阶段，与排队、抢占、训练代码和容器运行时无关。

## 复现环境

- qzcli 官方仓库：<https://github.com/tianyilt/qzcli_tool>
- Workspace：`分布式训练空间`
- Project：`高岳导师课程-具身智能机器人系统`
- Compute group ID：`lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7`
- Compute group：`开发区-H100-cuda12.8版本-183核`
- Spec ID：`8a53ac21-299a-4dee-85e9-9c04a544cf8d`
- 期望资源：单实例、4 张 H100
- Priority：`4`
- Framework：`pytorch`
- Image type：`SOURCE_PRIVATE`
- Image：`docker.sii.shaipower.online/inspire-studio/vegtea-dev-cc-conda-tailscale:1.5`
- Shared memory：`1200 GiB`

## MCP 最小复现参数

训练命令简化为 `echo` 即可复现，因为错误发生在任务创建阶段。

```json
{
  "workspace": "分布式训练空间",
  "project": "高岳导师课程-具身智能机器人系统",
  "compute_group": "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7",
  "spec": "8a53ac21-299a-4dee-85e9-9c04a544cf8d",
  "image": "docker.sii.shaipower.online/inspire-studio/vegtea-dev-cc-conda-tailscale:1.5",
  "image_type": "SOURCE_PRIVATE",
  "framework": "pytorch",
  "instances": 1,
  "shm": 1200,
  "priority": 4,
  "track": true,
  "name": "qzcli-h100-gpu-type-repro",
  "command": "echo qzcli-h100-gpu-type-repro"
}
```

## 实际结果

```text
API 请求失败: InvalidParameter: framework_config[0]: gpu_type "H100"
does not match logic_compute_group
"lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7"
real gpu type "NVIDIA_H100_SXM_80G"
```

使用计算组名称或 ID，以及不同 priority 时均出现同样错误。因此 priority、资源是否空闲、名称解析和训练入口不是触发条件。

## 预期结果

qzcli 应发送计算组要求的精确 GPU 枚举：

```text
NVIDIA_H100_SXM_80G
```

任务随后进入创建或排队状态，由平台按 priority 和资源情况调度。

## 初步源码定位

官方仓库 `qzcli/api.py` 的 `build_resource_spec_price()` 当前直接复制规格缓存里的 `gpu_type`：

```python
def build_resource_spec_price(
    spec_obj: Dict[str, Any], compute_group_id: str
) -> Dict[str, Any]:
    return {
        "cpu_type": "",
        "cpu_count": int(spec_obj.get("cpu_count") or 0),
        "gpu_type": spec_obj.get("gpu_type") or "",
        "gpu_count": int(spec_obj.get("gpu_count") or 0),
        "memory_size_gib": int(spec_obj.get("memory_gb") or 0),
        "logic_compute_group_id": compute_group_id,
        "quota_id": spec_obj.get("id") or "",
    }
```

本次规格缓存提供展示短名称 `H100`，而 `/api/v1/train_job/create` 校验平台精确枚举 `NVIDIA_H100_SXM_80G`，两者没有经过转换。

## 建议修复

长期方案是从 live quota/spec 或 compute-group 接口保留并透传平台返回的精确 GPU 类型，不要通过展示名称反推。

短期热修复应按计算组 ID 覆盖，避免把所有 `H100` 全局映射为 SXM 80G，从而误伤 H100 PCIe、NVL 或后续新增规格：

```python
_COMPUTE_GROUP_GPU_TYPE_OVERRIDES = {
    "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7": "NVIDIA_H100_SXM_80G",
}


def _platform_gpu_type(gpu_type: Any, compute_group_id: str) -> str:
    value = str(gpu_type or "").strip()
    return _COMPUTE_GROUP_GPU_TYPE_OVERRIDES.get(compute_group_id, value)
```

构造 payload 时使用：

```python
"gpu_type": _platform_gpu_type(
    spec_obj.get("gpu_type"), compute_group_id
),
```

## 建议回归测试

1. 目标计算组缓存值为 `H100` 时，payload 输出 `NVIDIA_H100_SXM_80G`。
2. 非目标计算组的未知 GPU 类型保持原值。
3. 缓存已经包含精确平台枚举时，不重复修改。
4. `--dry-run` 展示最终发送的 `resource_spec_price.gpu_type`。

## 临时绕过

修复 qzcli 前，可在启智网页端手动创建任务，并选择该计算组对应的精确 H100 SXM 80GB 规格。网页端 payload 通常来自 live 资源接口，不经过 qzcli 的短名称缓存。
