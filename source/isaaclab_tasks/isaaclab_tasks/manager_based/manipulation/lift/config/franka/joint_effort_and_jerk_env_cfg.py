# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# Config 3 — Energy-Aware Reward: Grasp Success + Joint Effort + Jerk Penalty
# Thesis: Energy-Aware Reward Shaping for Robotic Grasping
# Author: Su Acar, Tilburg University, 2026
# =============================================================================
#
# This configuration extends Config 2 by adding a joint acceleration penalty
# (jerk proxy) on top of the joint effort penalty. Joint acceleration is the
# rate of change of joint velocity, which serves as a proxy for motion
# smoothness. High accelerations correspond to jerky, energy-inefficient
# movements. Penalising this term encourages the robot to move more smoothly.
#
# Reward terms:
#   - reaching_object               (weight: +1.0)   [inherited from baseline]
#   - lifting_object                (weight: +15.0)  [inherited from baseline]
#   - object_goal_tracking          (weight: +16.0)  [inherited from baseline]
#   - object_goal_tracking_fine     (weight: +5.0)   [inherited from baseline]
#   - action_rate                   (weight: -1e-4)  [inherited, curriculum]
#   - joint_vel                     (weight: -1e-4)  [inherited, curriculum]
#   - joint_effort  (Config 2)      (weight: -1e-5)  [torque / energy penalty]
#   - joint_acc     (NEW)           (weight: -1e-5)  [jerk / smoothness penalty]
#weight is set lower than action_rate penalty to avoid hindering task completion, but still encourages energy-efficient motion
#franka has 7dof with large torque range, so we need a lower weight to prevent excessive penalties that could discourage exploration and learning
# =============================================================================

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.envs.mdp import rewards

from .joint_effort_env_cfg import (
    FrankaCubeLiftJointEffortEnvCfg,
    FrankaLiftJointEffortRewardsCfg,
)


# =============================================================================
# Step 1: Extend Config 2 rewards with the joint acceleration (jerk) penalty
# =============================================================================

@configclass
class FrankaLiftJointEffortAndJerkRewardsCfg(FrankaLiftJointEffortRewardsCfg):
    """Extends Config 2 rewards with a joint acceleration (jerk) penalty.

    Joint acceleration is the rate of change of joint velocity. High
    accelerations correspond to jerky, abrupt motions that are both
    mechanically stressful and energy-inefficient. Penalising this term
    on top of the torque penalty (Config 2) encourages the robot to
    adopt smoother, more energy-efficient trajectories.
    """

    joint_acc = RewTerm(
        func=rewards.joint_acc_l2,   # penalises L2 norm of joint accelerations
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# =============================================================================
# Step 2: Subclass FrankaCubeLiftJointEffortEnvCfg to use the new rewards
# =============================================================================

@configclass
class FrankaCubeLiftJointEffortAndJerkEnvCfg(FrankaCubeLiftJointEffortEnvCfg):
    """Franka lift environment with joint effort + jerk penalty (Config 3).

    Identical to Config 2 except the reward function now additionally
    includes a joint acceleration penalty to further encourage smooth,
    energy-efficient motion.
    """

    def __post_init__(self):
        # initialise everything from Config 2 first
        super().__post_init__()

        # replace the rewards with our effort + jerk version
        self.rewards = FrankaLiftJointEffortAndJerkRewardsCfg()


@configclass
class FrankaCubeLiftJointEffortAndJerkEnvCfg_PLAY(FrankaCubeLiftJointEffortAndJerkEnvCfg):
    """Play version of Config 3 — smaller scene, no randomisation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
