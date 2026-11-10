#!/usr/bin/env python3
"""
DQN-Explore-BC vs Deep Uniform Comparison Experiment

This script compares DQN-Explore-BC against Deep Uniform exploration baseline
on a 5x5 grid game instance.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import logging
from datetime import datetime
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from environments.grid_world import GridZeroSum, convert_gridzero_to_markov_game
from environments.player import Player
from game_solver import MarkovGameValueIteration
from additional_experiments.deep_mail import DeepWARMMAIL
from additional_experiments.utils import format_time
from additional_experiments.deep_unif_exploration import DeepUniform
from additional_experiments.utils import calc_exploitability_true_both

# # find cuda devices
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"  # specify which GPUs to use

def get_or_compute_expert_policies(num_states, num_a1, num_a2, rewards, transitions, gamma, logger):
    """
    Load expert policies from file if they exist, otherwise compute them via Value Iteration and save.
    
    Args:
        num_states: Number of states in the game
        num_a1: Number of actions for player 1
        num_a2: Number of actions for player 2
        rewards: Reward matrix
        transitions: Transition matrix
        gamma: Discount factor
        logger: Logger instance
        
    Returns:
        tuple: (mu_E_vi, nu_E_vi) - Expert policies for both players
    """
    # Create directory if it doesn't exist
    policy_dir = Path(__file__).resolve().parent / 'grid_expert_policies'
    policy_dir.mkdir(exist_ok=True)
    
    # Create a unique filename based on game parameters
    policy_filename = f'expert_policies_s{num_states}_a1_{num_a1}_a2_{num_a2}_gamma{gamma:.2f}.npz'
    policy_path = policy_dir / policy_filename
    
    # Try to load existing policies
    if policy_path.exists():
        logger.info(f"Loading expert policies from: {policy_path}")
        try:
            data = np.load(policy_path)
            mu_E_vi = data['mu_E_vi']
            nu_E_vi = data['nu_E_vi']
            logger.info("Expert policies loaded successfully from file")
            logger.info("")
            return mu_E_vi, nu_E_vi
        except Exception as e:
            logger.warning(f"Failed to load policies from file: {e}")
            logger.warning("Will recompute policies...")
    
    # Compute policies if not found or loading failed
    logger.info("Computing expert policies via Value Iteration...")
    expert_solver = MarkovGameValueIteration(num_states, num_a1, num_a2, rewards, transitions, discount_factor=gamma)
    _, mu_E_vi, nu_E_vi = expert_solver.value_iteration(max_iterations=1000)
    logger.info("Expert policies computed successfully")
    
    # Save policies to file
    try:
        np.savez(policy_path, mu_E_vi=mu_E_vi, nu_E_vi=nu_E_vi)
        logger.info(f"Expert policies saved to: {policy_path}")
    except Exception as e:
        logger.warning(f"Failed to save policies to file: {e}")
    
    logger.info("")
    return mu_E_vi, nu_E_vi


def setup_experiment_logging(log_file='DQN-Explore-BC_vs_deepunif.log'):
    """Set up logging for the comparison experiment."""
    log_dir = Path(__file__).resolve().parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'{timestamp}_{log_file}'
    
    logger = logging.getLogger('DQN-Explore-BC_vs_DeepUnif')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Experiment logging initialized. Log file: {log_path}")
    
    return logger, log_path, timestamp


def run_experiment():
    """Run the comparison experiment."""
    # Set up logging
    logger, log_path, timestamp = setup_experiment_logging()
    
    logger.info("="*80)
    logger.info("DQN-Explore-BC vs Deep Uniform Comparison Experiment")
    logger.info("="*80)
    
    # Experiment start time
    experiment_start_time = time.time()
    
    # =========================================================================
    # Game Setup
    # =========================================================================
    GAMMA = 0.9
    movements = ["left", "right", "up", "down"]
    grid_game = GridZeroSum(
        length=5, width=5,
        players=[Player(position=None, movements=movements), 
                 Player(position=None, movements=movements)],
        reward_coordinates=[[0, 4]], 
        reward_value=1, 
        gamma=GAMMA,
        starting_state=[[3, 0], [4, 1]]
    )
    
    REWARDS, TRANSITIONS, game_params = convert_gridzero_to_markov_game(grid_game)
    NUM_STATES = game_params['num_states']
    NUM_A1 = game_params['num_actions_p1']
    NUM_A2 = game_params['num_actions_p2']
    
    logger.info("Game Configuration:")
    logger.info("  Grid size: 5x5")
    logger.info(f"  Number of states: {NUM_STATES}")
    logger.info(f"  Player 1 actions: {NUM_A1}")
    logger.info(f"  Player 2 actions: {NUM_A2}")
    logger.info(f"  Gamma: {GAMMA}")
    logger.info("  Reward coordinates: [[0, 4]]")
    logger.info("")
    
    # Get or compute expert policies (will load from file if available)
    mu_E_vi, nu_E_vi = get_or_compute_expert_policies(
        num_states=NUM_STATES,
        num_a1=NUM_A1,
        num_a2=NUM_A2,
        rewards=REWARDS,
        transitions=TRANSITIONS,
        gamma=GAMMA,
        logger=logger
    )
    
    # Initial state distribution (uniform over all states)
    initial_state = [[3, 0], [4, 1]]
    initial_state_idx = grid_game.all_states.index(initial_state)
    initial_dist_c = np.zeros(NUM_STATES)
    initial_dist_c[initial_state_idx] = 1.0
    logger.info(f"Initial state distribution: {initial_dist_c}")
    logger.info("")
    

    # ========================================================================
    # Testing parameters
    # ========================================================================
    H = 16  # Horizon
    K = 500          # DQN-Explore-BC outer iterations
    K_UNIFORM = 1000  # Deep Uniform exploration trajectories per player
    DATASET_SIZES = [1, 50, 100, 500, 1000]  # Dataset sizes to test
    BC_EPOCHS = 1000
    SEEDS = [42, 123, 456]  # Random seeds for reproducibility
    BETA_VALUES = [0.1]  # Beta values to test
    TEMPERATURE_VALUES = [1.0]  # Temperature values to test 
    # DQN-Explore-BC specific parameters
    IMAGE_SIZE = 32
    DQN_HIDDEN_DIM = 128
    
    # Check available GPUs and assign them
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        DEVICE_MAIL = torch.device("cuda:0")
        logger.info(f"Found {num_gpus} GPUs. Using GPU 0 for DQN-Explore-BC")
    elif num_gpus == 1:
        DEVICE_MAIL = torch.device("cuda:0")
        logger.warning("Only 1 GPU found. Both experiments will share GPU 0")
    else:
        DEVICE_MAIL = torch.device("cpu")
        logger.warning("WARNING: No CUDA devices available. Training will be very slow on CPU!")
    
    logger.info("Experiment Parameters:")
    logger.info(f"  Horizon (H): {H}")
    logger.info(f"  K (max trajectories): {K}")
    logger.info(f"  Dataset sizes to test: {DATASET_SIZES}")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Beta values: {BETA_VALUES}")
    logger.info(f"  Temperature values: {TEMPERATURE_VALUES}")
    logger.info(f"  BC Epochs: {BC_EPOCHS}")
    logger.info(f"  DQN-Explore-BC Device: {DEVICE_MAIL}")
    logger.info(f"  Image Size (DQN-Explore-BC): {IMAGE_SIZE}")
    logger.info(f"  DQN Hidden Dim (DQN-Explore-BC): {DQN_HIDDEN_DIM}")
    logger.info(f"  Total experiments: {len(SEEDS)} seeds × {len(BETA_VALUES)} betas × {len(TEMPERATURE_VALUES)} temps × {len(DATASET_SIZES)} dataset sizes = {len(SEEDS) * len(BETA_VALUES) * len(TEMPERATURE_VALUES) * len(DATASET_SIZES)}")
    logger.info("")
    
    # =========================================================================
    # Run Experiments
    # =========================================================================
    results = {
        'seed': [],
        'dataset_size': [],
        'beta': [],
        'temperature': [],
        'num_samples': [],
        'deep_mail_exploitability': [],
        'deep_mail_time': [],
        'deep_uniform_exploitability': [],
        'deep_uniform_time': [],
    }
    
    total_experiments = len(SEEDS) * len(BETA_VALUES) * len(TEMPERATURE_VALUES) * len(DATASET_SIZES)
    experiment_counter = 0
    
    for seed in SEEDS:
        logger.info("\n" + "="*80)
        logger.info(f"SEED: {seed}")
        logger.info("="*80)

        # Flag for uniform experiment
        unif_not_done_for_seed = True

        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        for beta in BETA_VALUES:
            for temperature in TEMPERATURE_VALUES:
                logger.info("\n" + "="*80)
                logger.info(f"Running with Beta={beta}, Temp={temperature}, Seed={seed}")
                logger.info("="*80)
                
                # Run DQN-Explore-BC once with K=500, training BC on multiple dataset sizes
                logger.info(f"\n{'='*40}")
                logger.info(f"Running DQN-Explore-BC with K={K}, Beta={beta}, Temp={temperature}, Seed={seed} on {DEVICE_MAIL}")
                logger.info(f"{'='*40}")
                
                mail_start_time = time.time()
                
                deep_mail_solver = DeepWARMMAIL(
                    K=K,
                    num_states=NUM_STATES,
                    num_actions_p1=NUM_A1,
                    num_actions_p2=NUM_A2,
                    expert_policy_p1=mu_E_vi,
                    expert_policy_p2=nu_E_vi,
                    transition_P=TRANSITIONS,
                    initial_state_sampler=lambda: np.random.choice(NUM_STATES, p=initial_dist_c),
                    image_size=IMAGE_SIZE,
                    in_channels=3,
                    grid_game=grid_game,
                    dqn_hidden_dim=DQN_HIDDEN_DIM,
                    device=DEVICE_MAIL,
                    beta=beta,
                    temperature=temperature,
                    bc_cnn=True,
                    save_buffers=True
                )
                
                policies_mail, losses_mail, counts_mail = deep_mail_solver.run(
                    horizon=H,
                    epochs=BC_EPOCHS,
                    logger=logger,
                    dataset_sizes=DATASET_SIZES
                )
                
                mail_time = time.time() - mail_start_time
                logger.info(f"\nDQN-Explore-BC completed in {format_time(mail_time)}")

                # Uniform Experiment only needs to be run once per seed, there is no beta/temp dependence

                if unif_not_done_for_seed:
                    unif_not_done_for_seed = False
                    # Run Deep Uniform once with K=500, training BC on multiple dataset sizes
                    logger.info(f"\n{'='*40}")
                    logger.info(f"Running Deep Uniform with K={K_UNIFORM}, Seed={seed}")
                    logger.info(f"{'='*40}")
                    
                    uniform_start_time = time.time()
                    
                    deep_uniform_solver = DeepUniform(
                        K=K_UNIFORM,
                        num_states=NUM_STATES,
                        num_actions_p1=NUM_A1,
                        num_actions_p2=NUM_A2,
                        expert_policy_p1=mu_E_vi,
                        expert_policy_p2=nu_E_vi,
                        transition_P=TRANSITIONS,
                        initial_state_sampler=lambda: np.random.choice(NUM_STATES, p=initial_dist_c),
                        grid_game=grid_game,
                        gamma=GAMMA,
                        device=DEVICE_MAIL,
                        target_size=IMAGE_SIZE,
                        bc_cnn=True
                    )
                    
                    policies_uniform, losses_uniform = deep_uniform_solver.run(
                        horizon=H,
                        epochs=BC_EPOCHS,
                        logger=logger,
                        dataset_sizes=DATASET_SIZES
                    )
                    
                    uniform_time = time.time() - uniform_start_time
                    logger.info(f"\nDeep Uniform completed in {format_time(uniform_time)}")

                # Now evaluate each policy pair and store results
                for idx, dataset_size in enumerate(DATASET_SIZES):
                    experiment_counter += 1
                    logger.info(f"\n--- Evaluating dataset size {dataset_size} ({experiment_counter}/{total_experiments}) ---")
                    
                    # DQN-Explore-BC policy
                    hat_mu_mail, hat_nu_mail = policies_mail[idx]
                    exploitability_mail = calc_exploitability_true_both(
                        mu_pi=hat_mu_mail,
                        nu_pi=hat_nu_mail,
                        reward=REWARDS,
                        transition=TRANSITIONS,
                        initial_dist=initial_dist_c,
                        gamma=GAMMA
                    )
                    # counts_mail[idx] is a tuple (num_samples_p1, num_samples_p2)
                    num_samples_p1, num_samples_p2 = counts_mail[idx]
                    num_samples_mail = max(num_samples_p1, num_samples_p2)  # Use max for reporting
                    
                    # Deep Uniform policy
                    hat_mu_unif, hat_nu_unif = policies_uniform[idx]
                    exploitability_unif = calc_exploitability_true_both(
                        mu_pi=hat_mu_unif,
                        nu_pi=hat_nu_unif,
                        reward=REWARDS,
                        transition=TRANSITIONS,
                        initial_dist=initial_dist_c,
                        gamma=GAMMA
                    )
                    
                    
                    # Store results
                    results['seed'].append(seed)
                    results['dataset_size'].append(dataset_size)
                    results['beta'].append(beta)
                    results['temperature'].append(temperature)
                    results['num_samples'].append(num_samples_mail)
                    results['deep_mail_exploitability'].append(exploitability_mail)
                    results['deep_mail_time'].append(mail_time / len(DATASET_SIZES))  # Amortize time
                    results['deep_uniform_exploitability'].append(exploitability_unif)
                    results['deep_uniform_time'].append(uniform_time / len(DATASET_SIZES))  # Amortize time

                    # Save intermediate results after each evaluation
                    checkpoint_dir = Path(__file__).resolve().parent / 'results_all_algos'
                    checkpoint_dir.mkdir(exist_ok=True)

                    checkpoint_path = checkpoint_dir / f'DQN-Explore-BC_vs_deepunif_{timestamp}_partial.csv'
                    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                    

                    logger.info(f"  Dataset size: {dataset_size}")
                    logger.info(f"  DQN-Explore-BC exploitability:    {exploitability_mail:.6f}")
                    logger.info(f"  Deep Uniform exploitability: {exploitability_unif:.6f}")
                    logger.info(f"  DQN-Explore-BC improvement: {((exploitability_unif - exploitability_mail) / exploitability_unif * 100):.2f}%")
    
    experiment_total_time = time.time() - experiment_start_time
    
    # =========================================================================
    # Save Results
    # =========================================================================
    results_dir = Path(__file__).resolve().parent / 'results_all_algos'
    results_dir.mkdir(exist_ok=True)
    
    # Save CSV
    df = pd.DataFrame(results)
    csv_path = results_dir / f'DQN-Explore-BC_vs_deepunif_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to: {csv_path}")
    
    # =========================================================================
    # Generate Plots
    # =========================================================================
    logger.info("\nGenerating comparison plots...")
    
    # Convert results to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Group by dataset_size, beta, and temperature to compute mean and std across seeds
    grouped = df.groupby(['dataset_size', 'beta', 'temperature']).agg({
        'num_samples': 'mean',
        'deep_mail_exploitability': ['mean', 'std'],
        'deep_uniform_exploitability': ['mean', 'std'],
        'deep_mail_time': 'mean',
        'deep_uniform_time': 'mean',
    }).reset_index()
    
    # Create separate plots for each beta and temperature combination
    for beta in BETA_VALUES:
        for temp in TEMPERATURE_VALUES:
            subset = grouped[(grouped['beta'] == beta) & 
                           (grouped['temperature'] == temp)]
            
            if len(subset) == 0:
                continue
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            x_values = subset[('dataset_size', '')].values
            mail_mean = subset[('deep_mail_exploitability', 'mean')].values
            mail_std = subset[('deep_mail_exploitability', 'std')].values
            unif_mean = subset[('deep_uniform_exploitability', 'mean')].values
            unif_std = subset[('deep_uniform_exploitability', 'std')].values
            

            # Exploitability with error bars
            ax1.errorbar(x_values, mail_mean, yerr=mail_std, 
                        marker='o', linewidth=2, markersize=8, label='DQN-Explore-BC', 
                        color='blue', capsize=5)
            ax1.errorbar(x_values, unif_mean, yerr=unif_std,
                        marker='s', linewidth=2, markersize=8, label='Deep Uniform', 
                        color='red', capsize=5)
            
            ax1.set_xlabel('Dataset Size (Trajectories)', fontsize=12)
            ax1.set_ylabel('Exploitability', fontsize=12)
            ax1.set_title(f'Exploitability (β={beta}, T={temp})', fontsize=14)
            ax1.legend(fontsize=11)
            ax1.grid(True, which="both", ls="--", alpha=0.6)
            ax1.set_xscale('log')
            
            # Time comparison
            mail_time = subset[('deep_mail_time', 'mean')].values
            unif_time = subset[('deep_uniform_time', 'mean')].values
            
            ax2.plot(x_values, mail_time, 
                    marker='o', linewidth=2, markersize=8, label='DQN-Explore-BC', color='blue')
            ax2.plot(x_values, unif_time, 
                    marker='s', linewidth=2, markersize=8, label='Deep Uniform', color='red')
            
            ax2.set_xlabel('Dataset Size (Trajectories)', fontsize=12)
            ax2.set_ylabel('Amortized Training Time (seconds)', fontsize=12)
            ax2.set_title(f'Training Time (β={beta}, T={temp})', fontsize=14)
            ax2.legend(fontsize=11)
            ax2.grid(True, which="both", ls="--", alpha=0.6)
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            
            plt.tight_layout()
            plot_path = results_dir / f'comparison_beta{beta}_temp{temp}_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: {plot_path}")
            plt.close()
    
    # Create an overall summary plot averaging across all hyperparameters
    overall_grouped = df.groupby('dataset_size').agg({
        'num_samples': 'mean',
        'deep_mail_exploitability': ['mean', 'std'],
        'deep_uniform_exploitability': ['mean', 'std'],
        'deep_mail_time': 'mean',
        'deep_uniform_time': 'mean',
       
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x_values_overall = overall_grouped[('dataset_size', '')].values
    mail_mean_overall = overall_grouped[('deep_mail_exploitability', 'mean')].values
    mail_std_overall = overall_grouped[('deep_mail_exploitability', 'std')].values
    unif_mean_overall = overall_grouped[('deep_uniform_exploitability', 'mean')].values
    unif_std_overall = overall_grouped[('deep_uniform_exploitability', 'std')].values

    # Exploitability
    ax1.errorbar(x_values_overall, mail_mean_overall, yerr=mail_std_overall,
                marker='o', linewidth=2, markersize=8, label='DQN-Explore-BC', 
                color='blue', capsize=5)
    ax1.errorbar(x_values_overall, unif_mean_overall, yerr=unif_std_overall,
                marker='s', linewidth=2, markersize=8, label='Deep Uniform', 
                color='red', capsize=5)
    ax1.set_xlabel('Dataset Size (Trajectories)', fontsize=12)
    ax1.set_ylabel('Exploitability', fontsize=12)
    ax1.set_title('Overall Exploitability Comparison', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.set_xscale('log')
    ax1.set_xscale('log')
    
    # Time comparison
    mail_time_overall = overall_grouped[('deep_mail_time', 'mean')].values
    unif_time_overall = overall_grouped[('deep_uniform_time', 'mean')].values
    
    ax2.plot(x_values_overall, mail_time_overall, 
            marker='o', linewidth=2, markersize=8, label='DQN-Explore-BC', color='blue')
    ax2.plot(x_values_overall, unif_time_overall, 
            marker='s', linewidth=2, markersize=8, label='Deep Uniform', color='red')
    
    ax2.set_xlabel('Dataset Size (Trajectories)', fontsize=12)
    ax2.set_ylabel('Amortized Training Time (seconds)', fontsize=12)
    ax2.set_title('Overall Training Time Comparison', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plot_path = results_dir / f'all_algos_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Overall comparison plot saved to: {plot_path}")
    plt.close()
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Seed':<6} {'DatasetSz':<10} {'Beta':<8} {'Temp':<8} {'# Samples':<10} {'DQN-Explore-BC Exploit':<15} {'Unif Exploit':<15} {'DQN-Explore-BC Improv':<12}")
    logger.info("-" * 110)
    logger.info(f"\nTotal experiment time: {format_time(experiment_total_time)}")
    logger.info("")
    logger.info("="*80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Results CSV: {csv_path}")
    logger.info(f"Plots saved to: {results_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    run_experiment()
