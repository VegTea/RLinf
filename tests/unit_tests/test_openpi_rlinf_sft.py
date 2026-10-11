# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression checks for the additive OpenPI_RLinf SFT port.

Run: .venv-model312/bin/python -m pytest tests/unit_tests/test_openpi_rlinf_sft.py -q
Only network dimensions are reduced; the real SigLIP/Gemma/FM code is exercised.
"""

import os
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors.torch import save_file

from rlinf.config import SupportedModel, validate_fsdp_cfg
from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
    _validate_openpi_model_shape,
)
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy, get_lr_scheduler
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_rlinf import _apply_openpi_param_dtypes, get_model
from rlinf.models.embodiment.openpi_rlinf.modules import gemma, model, siglip
from rlinf.models.embodiment.openpi_rlinf.modules.utils import set_torch_compile
from rlinf.models.embodiment.openpi_rlinf.pi0 import Pi0
from rlinf.models.embodiment.openpi_rlinf.pi0_config import Pi0Config
from rlinf.runners.sft_runner import SFTRunner
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


def test_sft_step_and_checkpoint_roundtrip():
    torch.set_num_threads(2)
    set_torch_compile(False)
    vision = {
        "width": 32,
        "depth": 1,
        "num_heads": 4,
        "mlp_dim": 64,
        "patch_size": (56, 56),
    }
    with (
        patch.object(siglip, "_decode_variant", return_value=vision),
        patch.object(gemma, "PALIGEMMA_VOCAB_SIZE", 128),
        TemporaryDirectory() as tmp,
    ):
        cfg = Pi0Config(
            pi05=True,
            dtype="float32",
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_dim=4,
            action_horizon=2,
            max_token_len=4,
        )
        net = Pi0(cfg)
        obs = model.Observation(
            images={key: torch.zeros(1, 224, 224, 3) for key in model.IMAGE_KEYS},
            image_masks={
                key: torch.ones(1, dtype=torch.bool) for key in model.IMAGE_KEYS
            },
            state=torch.zeros(1, 4),
            tokenized_prompt=torch.zeros(1, 4, dtype=torch.long),
            tokenized_prompt_mask=torch.ones(1, 4, dtype=torch.bool),
        )
        batch = (obs, torch.randn(1, 2, 4))
        worker = SimpleNamespace(
            cfg=OmegaConf.create({"actor": {"model": {"model_type": "openpi_rlinf"}}}),
            amp_context=nullcontext(),
            model=net,
        )
        optim = torch.optim.AdamW(net.parameters(), lr=1e-3)
        before = net.action_out_proj.weight.detach().clone()
        loss = FSDPVlaSftWorker.get_train_model_output(worker, batch)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(net.action_out_proj.weight.grad).all()
        optim.step()
        assert not torch.equal(before, net.action_out_proj.weight)

        save_file(net.state_dict(), str(Path(tmp) / "model.safetensors"))
        factory_cfg = OmegaConf.create(
            {
                "model_type": "openpi_rlinf",
                "model_path": tmp,
                "precision": "fp32",
                "pi05": True,
                "action_dim": 4,
                "num_action_chunks": 2,
                "num_steps": 2,
                "openpi": {
                    "task": "sft",
                    "action_horizon": 2,
                    "model_action_dim": 4,
                    "paligemma_variant": "dummy",
                    "action_expert_variant": "dummy",
                    "max_token_len": 4,
                    "torch_compile": False,
                },
            }
        )
        loaded = get_model(factory_cfg)
        for key, value in net.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[key]), key
        torch.save(net.state_dict(), Path(tmp) / "full_weights.pt")
        (Path(tmp) / "model.safetensors").unlink()
        restored = get_model(factory_cfg)
        assert torch.equal(restored.action_out_proj.weight, net.action_out_proj.weight)
        torch.manual_seed(123)
        expected = net(forward_type=ForwardType.SFT, data=batch)
        torch.manual_seed(123)
        actual = restored(forward_type=ForwardType.SFT, data=batch)
        torch.testing.assert_close(actual, expected)
        with pytest.raises(ValueError, match="last dim"):
            restored.sft_forward((obs, torch.zeros(1, 2, 3)))


def test_schedule_shape_and_precision_guards():
    p = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.AdamW([p], lr=1.0)
    schedule = get_lr_scheduler("openpi_cosine", optim, 2, 6, min_lr=0.1)
    assert optim.param_groups[0]["lr"] == pytest.approx(1 / 3)
    rates = []
    for _ in range(6):
        optim.step()
        schedule.step()
        rates.append(optim.param_groups[0]["lr"])
    assert rates[1] == pytest.approx(1.0)
    assert rates[-1] == pytest.approx(0.1)

    official = SimpleNamespace(model=SimpleNamespace(action_horizon=50, action_dim=32))
    cfg = OmegaConf.create(
        {
            "num_action_chunks": 5,
            "openpi": {"model_action_dim": 32, "config_name": "pi0_aloha_robotwin"},
        }
    )
    _validate_openpi_model_shape(cfg, official)  # execution chunk != training horizon
    cfg.openpi.action_horizon = 5
    with pytest.raises(ValueError, match="different network"):
        _validate_openpi_model_shape(cfg, official)
    assert SupportedModel.OPENPI.value == "openpi"
    assert SupportedModel.OPENPI_RLINF.value == "openpi_rlinf"
    actor = OmegaConf.create(
        {
            "model": {"model_type": "openpi_rlinf"},
            "fsdp_config": {
                "strategy": "fsdp",
                "sharding_strategy": "no_shard",
                "use_orig_params": True,
                "mixed_precision": {
                    "param_dtype": None,
                    "reduce_dtype": None,
                    "buffer_dtype": None,
                },
            },
        }
    )
    validate_fsdp_cfg(actor)
    actor.fsdp_config.sharding_strategy = "full_shard"
    with pytest.raises(ValueError, match="no_shard"):
        validate_fsdp_cfg(actor)


def test_recipe_composition_and_step_limits():
    root = Path(__file__).resolve().parents[2]
    sft_dir = root / "examples/sft"
    with (
        patch.dict("os.environ", {"EMBODIED_PATH": str(sft_dir)}),
        initialize_config_dir(config_dir=str(sft_dir / "config"), version_base="1.1"),
    ):
        for name in (
            "robotwin_sft_openpi_rlinf",
            "behavior_sft_openpi_pi05_rlinf",
            "realworld_sft_openpi_pytorch_dual_franka_tcp_rot6d",
        ):
            cfg = compose(config_name=name)
            assert cfg.actor.model.model_type == "openpi_rlinf"
            assert cfg.actor.model.openpi.task == "sft"
            validate_fsdp_cfg(cfg.actor)
            assert cfg.actor.optim.lr_scheduler == "openpi_cosine"
        legacy = compose(config_name="robotwin_sft_openpi_pi05")
        assert legacy.actor.model.model_type == "openpi"
        validate_fsdp_cfg(legacy.actor)

    actor = Mock()
    actor.get_max_steps_per_epoch.return_value.wait.return_value = [5]
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "openpi_rlinf"}},
            "runner": {"max_epochs": -1, "max_steps": 13},
        }
    )
    runner = SimpleNamespace(cfg=cfg, actor=actor)
    SFTRunner.set_max_steps(runner)
    assert runner.max_steps == 13 and runner.num_steps_per_epoch == 5
    cfg.runner.max_epochs = 2
    SFTRunner.set_max_steps(runner)
    assert runner.max_steps == 10
    cfg.actor.model.model_type = "openpi"
    SFTRunner.set_max_steps(runner)
    assert runner.max_steps == 2 and runner.num_steps_per_epoch == 1


@pytest.mark.skipif(
    os.getenv("OPENPI_RLINF_CUDA_TEST") != "1",
    reason="Explicit CUDA smoke opt-in required",
)
def test_cuda_fsdp_sft_step():
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    torch.set_num_threads(2)
    set_torch_compile(False)
    vision = {
        "width": 32,
        "depth": 1,
        "num_heads": 4,
        "mlp_dim": 64,
        "patch_size": (56, 56),
    }
    with (
        TemporaryDirectory() as tmp,
        patch.object(siglip, "_decode_variant", return_value=vision),
        patch.object(gemma, "PALIGEMMA_VOCAB_SIZE", 128),
    ):
        dist.init_process_group(
            "nccl", init_method=f"file://{tmp}/pg", rank=0, world_size=1
        )
        try:
            net = Pi0(
                Pi0Config(
                    pi05=True,
                    dtype="bfloat16",
                    paligemma_variant="dummy",
                    action_expert_variant="dummy",
                    action_dim=4,
                    action_horizon=2,
                    max_token_len=4,
                )
            ).cuda()
            _apply_openpi_param_dtypes(net, None)
            wrapped = FSDP(
                net,
                use_orig_params=True,
                device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.NO_SHARD,
                auto_wrap_policy=get_fsdp_wrap_policy(net, model_type="openpi_rlinf"),
            )
            obs = model.Observation(
                images={key: torch.zeros(1, 224, 224, 3) for key in model.IMAGE_KEYS},
                image_masks={
                    key: torch.ones(1, dtype=torch.bool) for key in model.IMAGE_KEYS
                },
                state=torch.zeros(1, 4),
                tokenized_prompt=torch.zeros(1, 4, dtype=torch.long),
                tokenized_prompt_mask=torch.ones(1, 4, dtype=torch.bool),
            )
            optim = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
            before = net.action_out_proj.weight.detach().clone()
            loss = wrapped(
                forward_type=ForwardType.SFT, data=(obs, torch.randn(1, 2, 4))
            )
            assert torch.isfinite(loss)
            loss.backward()
            grad_norm = wrapped.clip_grad_norm_(1.0)
            assert torch.isfinite(grad_norm) and grad_norm > 0
            optim.step()
            assert not torch.equal(before, net.action_out_proj.weight)
            optim.zero_grad(set_to_none=True)
            weights = get_model_state_dict(
                wrapped, options=StateDictOptions(full_state_dict=True)
            )
            torch.save(
                {"model": weights, "optimizer": optim.state_dict()},
                Path(tmp) / "step.pt",
            )
            restored = torch.load(Path(tmp) / "step.pt", weights_only=True)
            set_model_state_dict(
                wrapped,
                restored["model"],
                options=StateDictOptions(full_state_dict=True),
            )
            optim.load_state_dict(restored["optimizer"])
            print(
                f"CUDA FSDP SFT loss={loss.item():.6f} grad_norm={grad_norm.item():.6f}; checkpoint restored"
            )
        finally:
            dist.destroy_process_group()
