# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RLinf registration for the rewarded Isaac Lab 3 Stack Cube task."""

import gymnasium as gym
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_env_cfg import (
    FrankaCubeStackVisuomotorEnvCfg,
)

RLINF_REWARDED_STACK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0"


@configclass
class RewardsCfg:
    """Sparse success reward used by the RLinf Stack Cube runs."""

    success = RewTerm(func=mdp.cubes_stacked, weight=20.0)


@configclass
class FrankaCubeStackVisuomotorRewardedEnvCfg(FrankaCubeStackVisuomotorEnvCfg):
    """Official Isaac Lab visuomotor task with RLinf's sparse reward."""

    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation


def register_rewarded_stack_env() -> None:
    """Register the RLinf task exactly once in Gymnasium."""
    if RLINF_REWARDED_STACK_ID in gym.registry:
        return
    gym.register(
        id=RLINF_REWARDED_STACK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": (
                "rlinf.envs.isaaclab.rewarded_stack_cfg:"
                "FrankaCubeStackVisuomotorRewardedEnvCfg"
            )
        },
        disable_env_checker=True,
    )
