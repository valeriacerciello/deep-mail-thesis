"""
Smoke test for DeepWARMMAIL.

Checks:
1. DeepWARMMAIL runs on the 5x5 gridworld.
2. Exploration trajectories have valid lengths.
3. Stored transitions agree with the environment dynamics.
4. Trajectories terminate immediately when the expert reaches the goal.
5. Exploration statistics are consistent.

Note:
If no expert reaches the goal during this small test, goal termination
cannot be conclusively validated.
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch

from environments.grid_world import (
    GridZeroSum,
    convert_gridzero_to_markov_game,
)
from environments.player import Player
from additional_experiments.deep_mail import DeepWARMMAIL
from game_solver import MarkovGameValueIteration


# ============================================================
# Configuration
# ============================================================

SEED = 42
GAMMA = 0.9

H = 16
K = 10
T_MAX = 3

BC_EPOCHS = 2
GRADIENT_STEPS = 1

DATASET_SIZES = [1, 5, 10]

IMAGE_SIZE = 32
DQN_HIDDEN_DIM = 128

DEVICE = torch.device("cpu")


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("DeepMAILSmokeTest")


# ============================================================
# Environment
# ============================================================

movements = ["left", "right", "up", "down"]

starting_state = [[3, 0], [4, 1]]

grid_game = GridZeroSum(
    length=5,
    width=5,
    players=[
        Player(position=None, movements=movements),
        Player(position=None, movements=movements),
    ],
    reward_coordinates=[[0, 4]],
    reward_value=1,
    gamma=GAMMA,
    starting_state=starting_state,
)

REWARDS, TRANSITIONS, game_params = (
    convert_gridzero_to_markov_game(grid_game)
)

NUM_STATES = game_params["num_states"]
NUM_A1 = game_params["num_actions_p1"]
NUM_A2 = game_params["num_actions_p2"]

initial_state_idx = grid_game.all_states.index(starting_state)

print(f"Number of states: {NUM_STATES}")
print(f"Actions per player: {NUM_A1}, {NUM_A2}")
print(f"Initial state index: {initial_state_idx}")


# ============================================================
# Expert policies
# ============================================================

expert_dir = Path("grid_expert_policies")
expert_dir.mkdir(exist_ok=True)

expert_path = expert_dir / (
    f"expert_policies_s{NUM_STATES}"
    f"_a1_{NUM_A1}"
    f"_a2_{NUM_A2}"
    f"_gamma{GAMMA:.2f}.npz"
)

if expert_path.exists():
    print(f"Loading expert policies from {expert_path}")

    expert_data = np.load(expert_path)

    print("Available expert-policy keys:", expert_data.files)

    mu_E_vi = expert_data["mu_E_vi"]
    nu_E_vi = expert_data["nu_E_vi"]

else:
    print("Computing expert policies...")

    vi_solver = MarkovGameValueIteration(
        num_states=NUM_STATES,
        num_actions_p1=NUM_A1,
        num_actions_p2=NUM_A2,
        reward=REWARDS,
        transition=TRANSITIONS,
        gamma=GAMMA,
    )

    mu_E_vi, nu_E_vi = vi_solver.value_iteration(
        max_iterations=1000
    )

    np.savez(
        expert_path,
        mu_E_vi=mu_E_vi,
        nu_E_vi=nu_E_vi,
    )

    print(f"Expert policies saved to {expert_path}")


# ============================================================
# Trajectory validation
# ============================================================

def validate_trajectories(buffer, exploring_player):
    """
    Validate trajectories collected during exploration.

    exploring_player:
        1 -> Player 1 explores, Player 2 is expert
        2 -> Player 2 explores, Player 1 is expert
    """

    trajectories = buffer.get_trajectories()

    lengths = [len(trajectory) for trajectory in trajectories]

    print(f"\nPlayer {exploring_player} exploration:")
    print(f"  Trajectories: {buffer.num_trajectories()}")
    print(f"  Transitions: {buffer.num_transitions()}")
    print(f"  Lengths: {lengths}")

    assert len(trajectories) > 0, (
        f"No trajectories collected for Player {exploring_player}"
    )

    assert all(0 < length <= H for length in lengths), (
        f"Invalid trajectory length for Player {exploring_player}"
    )

    assert sum(lengths) == buffer.num_transitions(), (
        f"Transition count mismatch for Player {exploring_player}"
    )

    goal_terminated = 0
    horizon_terminated = 0

    for trajectory_idx, trajectory in enumerate(trajectories):

        for step_idx, transition in enumerate(trajectory):

            state_idx = int(transition.state_idx)

            exploring_action = int(transition.action.item())
            expert_action = int(transition.expert_action.item())

            if exploring_player == 1:
                action_p1 = exploring_action
                action_p2 = expert_action
                expert_position_idx = 1
            else:
                action_p1 = expert_action
                action_p2 = exploring_action
                expert_position_idx = 0

            transition_probs = TRANSITIONS[
                state_idx,
                action_p1,
                action_p2,
            ]

            # This gridworld has deterministic transitions.
            assert np.isclose(
                transition_probs.sum(), 1.0
            ), "Transition probabilities do not sum to 1."

            assert np.isclose(
                transition_probs.max(), 1.0
            ), "Expected deterministic transitions."

            next_state_idx = int(np.argmax(transition_probs))

            next_state_coords = (
                grid_game.map_state_idx_to_state(next_state_idx)
            )

            expert_position = next_state_coords[
                expert_position_idx
            ]

            expert_reached_goal = (
                expert_position in grid_game.reward_coordinates
            )

            # A goal-reaching transition must be the last one.
            if expert_reached_goal:
                assert step_idx == len(trajectory) - 1, (
                    f"Player {exploring_player}, trajectory "
                    f"{trajectory_idx}: continued after expert "
                    f"reached the goal."
                )

            # Check that consecutive transitions connect correctly.
            if step_idx < len(trajectory) - 1:

                following_state_idx = int(
                    trajectory[step_idx + 1].state_idx
                )

                assert following_state_idx == next_state_idx, (
                    f"Player {exploring_player}, trajectory "
                    f"{trajectory_idx}: inconsistent state "
                    f"transition at step {step_idx}."
                )

        # Determine why this trajectory ended.
        if expert_reached_goal:
            goal_terminated += 1
        else:
            assert len(trajectory) == H, (
                f"Player {exploring_player}, trajectory "
                f"{trajectory_idx}: ended before horizon "
                f"without expert reaching the goal."
            )

            horizon_terminated += 1

    print(f"  Goal-terminated: {goal_terminated}")
    print(f"  Horizon-terminated: {horizon_terminated}")

    assert (
        goal_terminated + horizon_terminated
        == len(trajectories)
    )

    print(f"  Validation: PASSED")

    return goal_terminated


# ============================================================
# DeepWARMMAIL
# ============================================================

print("\nInitializing DeepWARMMAIL...")

solver = DeepWARMMAIL(
    K=K,
    num_states=NUM_STATES,
    num_actions_p1=NUM_A1,
    num_actions_p2=NUM_A2,
    expert_policy_p1=mu_E_vi,
    expert_policy_p2=nu_E_vi,
    transition_P=TRANSITIONS,
    initial_state_sampler=lambda: initial_state_idx,
    image_size=IMAGE_SIZE,
    in_channels=3,
    grid_game=grid_game,
    dqn_hidden_dim=DQN_HIDDEN_DIM,
    device=DEVICE,
    beta=0.1,
    temperature=1.0,
    bc_cnn=True,
    save_buffers=False,
    batch_size=8,
)


# ============================================================
# Run smoke test
# ============================================================

print("\nRunning DeepWARMMAIL smoke test...")

policies, losses, counts = solver.run(
    horizon=H,
    epochs=BC_EPOCHS,
    gradient_steps=GRADIENT_STEPS,
    t_max=T_MAX,
    logger=logger,
    dataset_sizes=DATASET_SIZES,
)


# ============================================================
# Validate exploration
# ============================================================

print("\nValidating exploration trajectories...")

p1_goal_terminated = validate_trajectories(
    solver.trajectory_buffer_p1,
    exploring_player=1,
)

p2_goal_terminated = validate_trajectories(
    solver.trajectory_buffer_p2,
    exploring_player=2,
)


# ============================================================
# Validate outputs
# ============================================================

assert len(policies) == len(DATASET_SIZES)
assert len(losses) == len(DATASET_SIZES)
assert len(counts) == len(DATASET_SIZES)

for policy_p1, policy_p2 in policies:

    assert policy_p1.shape == (NUM_STATES, NUM_A1)
    assert policy_p2.shape == (NUM_STATES, NUM_A2)

    assert np.all(np.isfinite(policy_p1))
    assert np.all(np.isfinite(policy_p2))

    assert np.allclose(policy_p1.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(policy_p2.sum(axis=1), 1.0, atol=1e-4)


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 60)
print("DEEPWARMMAIL SMOKE TEST COMPLETED")
print("=" * 60)

print(f"Player 1 trajectories: {solver.trajectory_buffer_p1.num_trajectories()}")
print(f"Player 2 trajectories: {solver.trajectory_buffer_p2.num_trajectories()}")

print(f"Player 1 goal terminations: {p1_goal_terminated}")
print(f"Player 2 goal terminations: {p2_goal_terminated}")

if p1_goal_terminated + p2_goal_terminated > 0:
    print("Goal-termination behavior: VALIDATED")
else:
    print(
        "Goal-termination behavior: NOT OBSERVED "
        "(additional testing required)"
    )

print("Trajectory consistency: PASSED")
print("Policy output validation: PASSED")
print("=" * 60)