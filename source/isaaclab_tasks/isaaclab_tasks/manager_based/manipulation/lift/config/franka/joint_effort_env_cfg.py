# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# Config 2 — Energy-Aware Reward: Grasp Success + Joint Effort Penalty
# Thesis: Energy-Aware Reward Shaping for Robotic Grasping
# Author: Su Acar, Tilburg University, 2026
# =============================================================================
#
# This configuration extends the baseline Franka lift environment by adding
# an explicit joint effort penalty to the reward function. Joint effort is
# computed as the L2 norm of applied joint torques, serving as a proxy for
# energy consumption. The robot is penalised for using high torques, which
# encourages more energy-efficient motion while still completing the task.
#
# Reward terms:
#   - reaching_object               (weight: +1.0)   [inherited from baseline]
#   - lifting_object                (weight: +15.0)  [inherited from baseline]
#   - object_goal_tracking          (weight: +16.0)  [inherited from baseline]
#   - object_goal_tracking_fine     (weight: +5.0)   [inherited from baseline]
#   - action_rate                   (weight: -1e-4)  [inherited, curriculum]
#   - joint_vel                     (weight: -1e-4)  [inherited, curriculum]
#   - joint_effort  (NEW)           (weight: -1e-5)  [energy penalty]
#weight is set lower than action_rate penalty to avoid hindering task completion, but still encourages energy-efficient motion
#franka has 7dof with large torque range, so we need a lower weight to prevent excessive penalties that could discourage exploration and learning
# =============================================================================

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.envs.mdp import rewards
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import RewardsCfg

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg, FrankaCubeLiftEnvCfg_PLAY


# =============================================================================
# Step 1: Subclass RewardsCfg to add the joint effort penalty
# =============================================================================

@configclass
class FrankaLiftJointEffortRewardsCfg(RewardsCfg):
    """Extends the baseline rewards with a joint effort (torque) penalty.

    Joint effort is the L2 norm of applied joint torques across all robot
    joints. Higher torques generally correspond to higher energy consumption,
    so penalising this term encourages the robot to use lighter, more
    energy-efficient movements.
    """

    joint_effort = RewTerm(
        func=rewards.joint_torques_l2,  # penalises L2 norm of applied joint torques
        #weight is set lower than action_rate penalty to avoid hindering task completion, but still encourages energy-efficient motion
        #franka has 7dof with large torque range, so we need a lower weight to prevent excessive penalties that could discourage exploration and learning
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# =============================================================================
# Step 2: Subclass FrankaCubeLiftEnvCfg to use the new rewards
# =============================================================================

@configclass
class FrankaCubeLiftJointEffortEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka lift environment with joint effort penalty.

    Identical to the baseline (Config 1) except the reward function now
    includes an explicit joint effort penalty term to encourage
    energy-efficient motion.
    """

    def __post_init__(self):
        # initialise everything from the baseline config first
        super().__post_init__()

        # replace the rewards with our energy-aware version
        self.rewards = FrankaLiftJointEffortRewardsCfg()


@configclass
class FrankaCubeLiftJointEffortEnvCfg_PLAY(FrankaCubeLiftJointEffortEnvCfg):
    """Play version of Config 2 — smaller scene, no randomisation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False