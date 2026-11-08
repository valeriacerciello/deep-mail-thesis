#!/usr/bin/env python3
"""Lightweight DQN-Explore-BC vs Deep Uniform comparison on a 5x5 grid.

Reports exploitability, expert-labelled BC transitions, exploration interactions,
and full algorithm runtime. Equal K is NOT an equal interaction budget.
"""
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environments.grid_world import GridZeroSum, convert_gridzero_to_markov_game
from environments.player import Player
from game_solver import MarkovGameValueIteration
from additional_experiments.deep_mail import DeepWARMMAIL
from additional_experiments.deep_unif_exploration import DeepUniform
from additional_experiments.utils import format_time, calc_exploitability_true_both

# Smoke-test settings only; not a statistically meaningful benchmark.
GAMMA = 0.9
H = 16
K = 10
T_MAX = 3
GRADIENT_STEPS = 1
DQN_BATCH_SIZE = 8
DATASET_SIZES = [1, 5, 10]
BC_EPOCHS = 2
SEEDS = [42]
BETA_VALUES = [0.1]
TEMPERATURE_VALUES = [1.0]
IMAGE_SIZE = 32
DQN_HIDDEN_DIM = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging():
    root = Path(__file__).resolve().parent
    folder = root / 'logs'
    folder.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = folder / f'{timestamp}_DQN-Explore-BC_vs_deepunif.log'
    logger = logging.getLogger('DQN-Explore-BC_vs_DeepUnif')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in (logging.FileHandler(log_path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, log_path, timestamp


def get_or_compute_expert_policies(num_states, num_a1, num_a2, rewards, transitions, gamma, logger):
    folder = Path(__file__).resolve().parent / 'grid_expert_policies'
    folder.mkdir(exist_ok=True)
    path = folder / f'expert_policies_s{num_states}_a1_{num_a1}_a2_{num_a2}_gamma{gamma:.2f}.npz'
    if path.exists():
        try:
            with np.load(path) as data:
                mu, nu = data['mu_E_vi'], data['nu_E_vi']
            assert mu.shape == (num_states, num_a1)
            assert nu.shape == (num_states, num_a2)
            logger.info('Loaded expert policies from %s', path)
            return mu, nu
        except (KeyError, ValueError, AssertionError, OSError) as exc:
            logger.warning('Cannot load cached experts (%s); recomputing.', exc)
    logger.info('Computing expert policies with value iteration...')
    solver = MarkovGameValueIteration(num_states, num_a1, num_a2, rewards, transitions,
                                      discount_factor=gamma)
    _, mu, nu = solver.value_iteration(max_iterations=1000)
    np.savez(path, mu_E_vi=mu, nu_E_vi=nu)
    logger.info('Saved expert policies to %s', path)
    return mu, nu


def bc_samples(buffer, size):
    """Number of expert-labelled transitions in the first `size` trajectories."""
    trajectories = buffer.get_trajectories()
    if len(trajectories) < size:
        raise ValueError(f'Requested {size} BC trajectories, but only {len(trajectories)} exist')
    return sum(len(trajectory) for trajectory in trajectories[:size])


def exploration_stats(solver, prefix):
    p1 = solver.trajectory_buffer_p1  # Player 1 explores; Player 2 supplies labels.
    p2 = solver.trajectory_buffer_p2  # Player 2 explores; Player 1 supplies labels.
    return {
        f'{prefix}_p1_trajectories': p1.num_trajectories(),
        f'{prefix}_p2_trajectories': p2.num_trajectories(),
        f'{prefix}_p1_transitions': p1.num_transitions(),
        f'{prefix}_p2_transitions': p2.num_transitions(),
        f'{prefix}_total_transitions': p1.num_transitions() + p2.num_transitions(),
    }


def evaluate_exploitability(policy_pair, rewards, transitions, initial_dist):
    mu, nu = policy_pair
    return float(calc_exploitability_true_both(
        mu_pi=mu, nu_pi=nu, reward=rewards, transition=transitions,
        initial_dist=initial_dist, gamma=GAMMA))


def plot_comparison(frame, title, path, logger):
    """Original two-panel visualization: exploitability and amortized runtime."""
    grouped = frame.groupby('dataset_size', as_index=False).agg(
        mail_mean=('deep_mail_exploitability', 'mean'),
        mail_std=('deep_mail_exploitability', 'std'),
        uniform_mean=('deep_uniform_exploitability', 'mean'),
        uniform_std=('deep_uniform_exploitability', 'std'),
        mail_time=('deep_mail_time', 'mean'),
        uniform_time=('deep_uniform_time', 'mean'),
    ).fillna(0)
    x = grouped['dataset_size'].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for label, mean, std, marker, color in (
        ('DQN-Explore-BC', 'mail_mean', 'mail_std', 'o', 'blue'),
        ('Deep Uniform', 'uniform_mean', 'uniform_std', 's', 'red'),
    ):
        ax1.errorbar(x, grouped[mean], yerr=grouped[std], marker=marker,
                     linewidth=2, markersize=8, capsize=5, label=label, color=color)
    ax1.set_xlabel('Dataset Size (Trajectories)')
    ax1.set_ylabel('Exploitability')
    ax1.set_title(f'{title}: Exploitability')
    ax1.set_xscale('log')
    ax1.grid(True, ls='--', alpha=.6)
    ax1.legend()
    for label, column, marker, color in (
        ('DQN-Explore-BC', 'mail_time', 'o', 'blue'),
        ('Deep Uniform', 'uniform_time', 's', 'red'),
    ):
        ax2.plot(x, grouped[column], marker=marker, linewidth=2, markersize=8,
                 label=label, color=color)
    ax2.set_xlabel('Dataset Size (Trajectories)')
    ax2.set_ylabel('Amortized Total Runtime (seconds)')
    ax2.set_title(f'{title}: Runtime')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, ls='--', alpha=.6)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info('Plot saved to %s', path)


def run_experiment():
    logger, log_path, timestamp = setup_logging()
    experiment_start = time.time()
    logger.info('DQN-Explore-BC vs Deep Uniform | CPU smoke comparison')
    logger.info('H=%s K=%s dataset_sizes=%s BC_epochs=%s', H, K, DATASET_SIZES, BC_EPOCHS)
    logger.info('DQN gradient_steps=%s t_max=%s batch_size=%s',
                GRADIENT_STEPS, T_MAX, DQN_BATCH_SIZE)
    logger.warning('Equal K does not imply equal trajectories, interactions or BC samples.')

    movements = ['left', 'right', 'up', 'down']
    initial_state = [[3, 0], [4, 1]]
    grid_game = GridZeroSum(
        length=5, width=5,
        players=[Player(position=None, movements=movements),
                 Player(position=None, movements=movements)],
        reward_coordinates=[[0, 4]], reward_value=1,
        gamma=GAMMA, starting_state=initial_state,
    )
    rewards, transitions, game_params = convert_gridzero_to_markov_game(grid_game)
    num_states = game_params['num_states']
    num_a1 = game_params['num_actions_p1']
    num_a2 = game_params['num_actions_p2']
    mu_expert, nu_expert = get_or_compute_expert_policies(
        num_states, num_a1, num_a2, rewards, transitions, GAMMA, logger)
    initial_state_idx = grid_game.all_states.index(initial_state)
    initial_dist = np.zeros(num_states)
    initial_dist[initial_state_idx] = 1.0
    initial_sampler = lambda: initial_state_idx
    logger.info('Grid states=%d, actions=(%d,%d), initial_state_idx=%d',
                num_states, num_a1, num_a2, initial_state_idx)

    rows = []
    for seed in SEEDS:
        # Uniform is independent of beta and temperature: run once per seed.
        seed_everything(seed)
        logger.info('Running Deep Uniform, seed=%s', seed)
        uniform_start = time.perf_counter()
        uniform_solver = DeepUniform(
            K=K, num_states=num_states, num_actions_p1=num_a1,
            num_actions_p2=num_a2, expert_policy_p1=mu_expert,
            expert_policy_p2=nu_expert, transition_P=transitions,
            initial_state_sampler=initial_sampler, grid_game=grid_game,
            gamma=GAMMA, device=DEVICE, target_size=IMAGE_SIZE, bc_cnn=True)
        uniform_policies, uniform_losses = uniform_solver.run(
            horizon=H, epochs=BC_EPOCHS, logger=logger, dataset_sizes=DATASET_SIZES)
        uniform_time = time.perf_counter() - uniform_start
        uniform_stats = exploration_stats(uniform_solver, 'deep_uniform')
        logger.info('Uniform: %s, runtime=%.3fs', uniform_stats, uniform_time)
        uniform_exploitabilities = [evaluate_exploitability(p, rewards, transitions, initial_dist)
                                    for p in uniform_policies]

        for beta in BETA_VALUES:
            for temperature in TEMPERATURE_VALUES:
                seed_everything(seed)  # Independent, reproducible DQN run.
                logger.info('Running DQN-Explore-BC, seed=%s beta=%s temperature=%s',
                            seed, beta, temperature)
                mail_start = time.perf_counter()
                mail_solver = DeepWARMMAIL(
                    K=K, num_states=num_states, num_actions_p1=num_a1,
                    num_actions_p2=num_a2, expert_policy_p1=mu_expert,
                    expert_policy_p2=nu_expert, transition_P=transitions,
                    initial_state_sampler=initial_sampler,
                    image_size=IMAGE_SIZE, in_channels=3, grid_game=grid_game,
                    dqn_hidden_dim=DQN_HIDDEN_DIM, device=DEVICE,
                    beta=beta, temperature=temperature, bc_cnn=True,
                    save_buffers=False, batch_size=DQN_BATCH_SIZE)
                mail_policies, mail_losses, mail_counts = mail_solver.run(
                    horizon=H, epochs=BC_EPOCHS, gradient_steps=GRADIENT_STEPS,
                    t_max=T_MAX, logger=logger, dataset_sizes=DATASET_SIZES)
                mail_time = time.perf_counter() - mail_start
                mail_stats = exploration_stats(mail_solver, 'deep_mail')
                logger.info('DQN: %s, runtime=%.3fs', mail_stats, mail_time)

                if not (len(mail_policies) == len(uniform_policies) == len(DATASET_SIZES)):
                    raise RuntimeError('BC policy count differs from DATASET_SIZES')
                for idx, size in enumerate(DATASET_SIZES):
                    # Buffer p2 contains Player 1 expert labels; buffer p1 contains Player 2 labels.
                    mail_p1_samples = bc_samples(mail_solver.trajectory_buffer_p2, size)
                    mail_p2_samples = bc_samples(mail_solver.trajectory_buffer_p1, size)
                    uniform_p1_samples = bc_samples(uniform_solver.trajectory_buffer_p2, size)
                    uniform_p2_samples = bc_samples(uniform_solver.trajectory_buffer_p1, size)
                    if tuple(mail_counts[idx]) != (mail_p1_samples, mail_p2_samples):
                        logger.warning('DQN returned counts %s differ from buffer-derived BC counts (%s,%s)',
                                       mail_counts[idx], mail_p1_samples, mail_p2_samples)
                    mail_exploit = evaluate_exploitability(mail_policies[idx], rewards, transitions, initial_dist)
                    uniform_exploit = uniform_exploitabilities[idx]
                    improvement = (100 * (uniform_exploit - mail_exploit) / uniform_exploit
                                   if abs(uniform_exploit) > 1e-12 else float('nan'))
                    row = {
                        'seed': seed, 'dataset_size': size, 'beta': beta,
                        'temperature': temperature,
                        # Backward-compatible legacy field: max DQN BC sample count.
                        'num_samples': max(mail_p1_samples, mail_p2_samples),
                        'deep_mail_exploitability': mail_exploit,
                        'deep_uniform_exploitability': uniform_exploit,
                        # Preserve old amortized time columns for existing plots.
                        'deep_mail_time': mail_time / len(DATASET_SIZES),
                        'deep_uniform_time': uniform_time / len(DATASET_SIZES),
                        'deep_mail_total_runtime': mail_time,
                        'deep_uniform_total_runtime': uniform_time,
                        'deep_mail_p1_bc_samples': mail_p1_samples,
                        'deep_mail_p2_bc_samples': mail_p2_samples,
                        'deep_uniform_p1_bc_samples': uniform_p1_samples,
                        'deep_uniform_p2_bc_samples': uniform_p2_samples,
                        'dqn_improvement_pct': improvement,
                        'horizon': H, 'k': K, 'bc_epochs': BC_EPOCHS,
                        'dqn_gradient_steps': GRADIENT_STEPS, 'dqn_t_max': T_MAX,
                        'dqn_batch_size': DQN_BATCH_SIZE,
                        **mail_stats, **uniform_stats,
                    }
                    rows.append(row)
                    logger.info('seed=%s size=%s | DQN exploit=%.6f (BC labels P1=%d P2=%d) '
                                '| Uniform exploit=%.6f (BC labels P1=%d P2=%d) | improvement=%s',
                                seed, size, mail_exploit, mail_p1_samples, mail_p2_samples,
                                uniform_exploit, uniform_p1_samples, uniform_p2_samples,
                                f'{improvement:.2f}%' if np.isfinite(improvement) else 'undefined')

    results_dir = Path(__file__).resolve().parent / 'results_all_algos'
    results_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = results_dir / f'DQN-Explore-BC_vs_deepunif_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    logger.info('CSV saved: %s', csv_path)
    for beta in BETA_VALUES:
        for temp in TEMPERATURE_VALUES:
            subset = df[(df['beta'] == beta) & (df['temperature'] == temp)]
            if not subset.empty:
                plot_comparison(subset, f'β={beta}, T={temp}',
                                results_dir / f'comparison_beta{beta}_temp{temp}_{timestamp}.png', logger)
    plot_comparison(df, 'Overall Comparison', results_dir / f'all_algos_{timestamp}.png', logger)
    logger.info('FINAL RESULTS SUMMARY\n%s', df[
        ['seed', 'dataset_size', 'beta', 'temperature',
         'deep_mail_exploitability', 'deep_uniform_exploitability',
         'deep_mail_p1_bc_samples', 'deep_mail_p2_bc_samples',
         'deep_uniform_p1_bc_samples', 'deep_uniform_p2_bc_samples']
    ].to_string(index=False))
    logger.info('Total experiment time: %s', format_time(time.time() - experiment_start))
    logger.info('Experiment completed. Log: %s | CSV: %s', log_path, csv_path)
    return df


if __name__ == '__main__':
    run_experiment()
