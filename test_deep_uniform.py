import sys
from pathlib import Path

import numpy as np
import torch
import logging

# Add the repository root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from environments.grid_world import GridZeroSum, convert_gridzero_to_markov_game
from environments.player import Player
from game_solver import MarkovGameValueIteration
from additional_experiments.deep_unif_exploration import DeepUniform

# ============================================================
# Test configuration
# ============================================================

H = 16
K = 10
GAMMA = 0.9

np.random.seed(42)
torch.manual_seed(42)

logger = logging.getLogger("DeepUniformTest")
logging.basicConfig(level=logging.INFO)

# ============================================================
# Create the gridworld
# ============================================================

movements = ["left", "right", "up", "down"]

grid_game = GridZeroSum(
    length=5,
    width=5,
    players=[
        Player(position=None, movements=movements),
        Player(position=None, movements=movements)
    ],
    reward_coordinates=[[0, 4]],
    reward_value=1,
    gamma=GAMMA,
    starting_state=[[3, 0], [4, 1]]
)

REWARDS, TRANSITIONS, game_params = convert_gridzero_to_markov_game(grid_game)

NUM_STATES = game_params["num_states"]
NUM_A1 = game_params["num_actions_p1"]
NUM_A2 = game_params["num_actions_p2"]

print(f"Number of states: {NUM_STATES}")
print(f"Number of actions: Player 1 = {NUM_A1}, Player 2 = {NUM_A2}")

# ============================================================
# Load or compute expert policies
# ============================================================

policy_dir = Path(__file__).resolve().parent / "grid_expert_policies"
policy_dir.mkdir(exist_ok=True)

policy_path = policy_dir / (
    f"expert_policies_s{NUM_STATES}_a1_{NUM_A1}"
    f"_a2_{NUM_A2}_gamma{GAMMA:.2f}.npz"
)

if policy_path.exists():
    print("Loading cached expert policies...")

    data = np.load(policy_path)
    mu_E_vi = data["mu_E_vi"]
    nu_E_vi = data["nu_E_vi"]

else:
    print("Computing expert policies...")

    expert_solver = MarkovGameValueIteration(
        NUM_STATES,
        NUM_A1,
        NUM_A2,
        REWARDS,
        TRANSITIONS,
        discount_factor=GAMMA
    )

    _, mu_E_vi, nu_E_vi = expert_solver.value_iteration(
        max_iterations=1000
    )

    np.savez(
        policy_path,
        mu_E_vi=mu_E_vi,
        nu_E_vi=nu_E_vi
    )

    print("Expert policies computed and saved!")

print(f"Player 1 expert policy shape: {mu_E_vi.shape}")
print(f"Player 2 expert policy shape: {nu_E_vi.shape}")

# ============================================================
# Initialize Deep Uniform
# ============================================================

initial_state = [[3, 0], [4, 1]]
initial_state_idx = grid_game.all_states.index(initial_state)

initial_dist = np.zeros(NUM_STATES)
initial_dist[initial_state_idx] = 1.0

deep_uniform_solver = DeepUniform(
    K=K,
    num_states=NUM_STATES,
    num_actions_p1=NUM_A1,
    num_actions_p2=NUM_A2,
    expert_policy_p1=mu_E_vi,
    expert_policy_p2=nu_E_vi,
    transition_P=TRANSITIONS,
    initial_state_sampler=lambda: np.random.choice(
        NUM_STATES, p=initial_dist
    ),
    grid_game=grid_game,
    gamma=GAMMA,
    device=torch.device("cpu"),
    target_size=32,
    bc_cnn=True
)

print("Deep Uniform initialized successfully!")

# ============================================================
# Run Deep Uniform
# ============================================================

print("\nStarting Deep Uniform exploration...")

policies, losses = deep_uniform_solver.run(
    horizon=H,
    epochs=2,
    logger=logger,
    dataset_sizes=[1, 5, 10]
)

# ============================================================
# Verify collected trajectories
# ============================================================

buffer_p1 = deep_uniform_solver.trajectory_buffer_p1
buffer_p2 = deep_uniform_solver.trajectory_buffer_p2

for player, buffer in [(1, buffer_p1), (2, buffer_p2)]:
    trajectories = buffer.get_trajectories()
    lengths = [len(trajectory) for trajectory in trajectories]

    print(f"\nPlayer {player}:")
    print(f"  Number of trajectories: {len(trajectories)}")
    print(f"  Total transitions: {buffer.num_transitions()}")
    print(f"  Trajectory lengths: {lengths}")

    assert len(trajectories) == K, (
        f"Player {player}: Expected {K} trajectories, "
        f"got {len(trajectories)}"
    )

    assert all(1 <= length <= H for length in lengths), (
        f"Player {player}: Invalid trajectory length detected"
    )

print("\nAll trajectory checks passed!")

