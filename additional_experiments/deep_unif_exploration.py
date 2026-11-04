# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Use free GPUs

import torch
import numpy as np
from tqdm import tqdm
import logging
import time
from torchvision import transforms

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from additional_experiments.bc import BehavioralCloningSingleAgent
from additional_experiments.utils import TrajectoryBuffer, format_time


class DeepUniform:
    def __init__(
        self,
        K,
        num_states,
        num_actions_p1,
        num_actions_p2,
        expert_policy_p1,
        expert_policy_p2,
        transition_P,
        initial_state_sampler,
        grid_game,
        gamma=0.9,
        bc_cnn=False,
        device='cpu',
        target_size=None
    ):
        self.K = K
        self.num_states = num_states
        self.num_actions_p1 = num_actions_p1
        self.num_actions_p2 = num_actions_p2
        self.expert_policy_p1 = expert_policy_p1
        self.expert_policy_p2 = expert_policy_p2
        self.transition_P = transition_P
        self.initial_state_sampler = initial_state_sampler
        self.gamma = gamma
        self.grid_game = grid_game
        self.bc_cnn = bc_cnn
        self.device = device
        self.target_size = target_size

        # Trajectory buffers for both players (without capacity, stores all exploration data)
        self.trajectory_buffer_p1 = TrajectoryBuffer(player_id=1)
        self.trajectory_buffer_p2 = TrajectoryBuffer(player_id=2)

    def run(self, horizon, epochs=1000, logger=None, dataset_sizes=None, max_transitions=None):
        """Main training loop for Deep Uniform.
        
        Args:
            horizon: Episode horizon
            epochs: Number of BC training epochs
            logger: Logger instance
            dataset_sizes: List of dataset sizes to train BC on (e.g., [1, 50, 100, 500]).
                          If None, trains once on all collected data.
            max_transitions: Maximum number of transitions to collect. If reached, exploration stops.
        
        Returns:
            If dataset_sizes is None:
                ((hat_mu, hat_nu), (loss_mu, loss_nu))
            If dataset_sizes is provided:
                (policies_list, losses_list)
                where each element corresponds to a dataset size
        """
        if logger is None:
            logger = logging.getLogger('DeepUniform')

        
        logger.info("="*80)
        logger.info("Starting Deep Uniform")
        logger.info("="*80)
        logger.info("Configuration:")
        logger.info(f"  K (Uniform iterations): {self.K}")
        logger.info(f"  Horizon: {horizon}")
        logger.info(f"  Gamma (discount): {self.gamma}")
        if dataset_sizes is not None:
            logger.info(f"  Dataset sizes for BC: {dataset_sizes}")
        logger.info("")
        
        # run exploration for the first player
        logger.info("="*80)
        logger.info("PHASE 1: Exploring with Player 1")
        logger.info("="*80)
    
        
        for k in tqdm(range(self.K)):
            # in this k is the index of a trajectory that we collect while player 1 plays uniformly and player 2 plays expert
            logger.info(f"\n--- Player 1: Iteration {k+1}/{self.K} ---")
            # sample the trajectory
            state = self.initial_state_sampler()
            for _ in range(horizon):
                action_p1_probs = np.ones(self.num_actions_p1) / self.num_actions_p1
                action_p1 = np.random.choice(self.num_actions_p1, p=action_p1_probs)
                action_p2 = np.random.choice(self.num_actions_p2, p=self.expert_policy_p2[state])
                next_state = np.random.choice(self.num_states, p=self.transition_P[state, action_p1, action_p2])
                
                # Check if expert player (Player 2) reached the reward cell
                next_state_coords = self.grid_game.map_state_idx_to_state(next_state)
                player2_pos = next_state_coords[1]  # Player 2's position
                

                # Store transition in trajectory buffer (for BC training later)
                if self.bc_cnn:
                    img_array = self.grid_game.render(state=state)
                    img_array = img_array.astype(np.float32) / 255.0  # Normalize pixel values
                    state_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                    if self.target_size is not None:
                        resize_transform = transforms.Resize(self.target_size, antialias=True)
                        state_tensor = resize_transform(state_tensor)
                    next_img_array = self.grid_game.render(state=next_state)
                    next_img_array = next_img_array.astype(np.float32) / 255.0
                    next_state_tensor = torch.from_numpy(next_img_array).permute(2, 0, 1)
                    if self.target_size is not None:
                        resize_transform = transforms.Resize(self.target_size, antialias=True)
                        next_state_tensor = resize_transform(next_state_tensor)
                    
                    self.trajectory_buffer_p1.push(
                        state_tensor,
                        action_p1,
                        action_p2,  # Expert action is from p2
                        next_state_tensor,
                    )
                else:       
                    self.trajectory_buffer_p1.push(
                        state,
                        action_p1,
                        action_p2,
                        next_state,
                    )

                state = next_state

                # Stop only the current trajectory when the expert reaches its goal
                if player2_pos in self.grid_game.reward_coordinates:
                    logger.info(
                        f"Player 1 exploration, trajectory {k+1}: "
                        f"Expert Player 2 reached goal at {player2_pos}. "
                        f"Terminating current trajectory."
                    )
                    break

            # End the trajectory in the trajectory buffer
            self.trajectory_buffer_p1.end_trajectory()



            #     state = next_state
                
            # # End the trajectory in the trajectory buffer
            # self.trajectory_buffer_p1.end_trajectory()

            # if player2_pos in self.grid_game.reward_coordinates:
            #     break 

            if max_transitions is not None and self.trajectory_buffer_p1.num_transitions() >= max_transitions:
                logger.info(f"Max transitions reached for Player 1: {self.trajectory_buffer_p1.num_transitions()} >= {max_transitions}")
                break


        logger.info(
            f"Player 1 exploration completed: "
            f"{self.trajectory_buffer_p1.num_trajectories()} trajectories, "
            f"{self.trajectory_buffer_p1.num_transitions()} transitions"
        )
        
        # run exploration for the second player
        logger.info("="*80)
        logger.info("PHASE 2: Exploring with Player 2")
        logger.info("="*80)
        
        for k in tqdm(range(self.K)):
            logger.info(f"\n--- Player 2: Iteration {k+1}/{self.K} ---")
            state = self.initial_state_sampler()

            for _ in range(horizon):
                # Player 1 uses the expert policy this time and player 2 plays uniformly
                action_p1 = np.random.choice(self.num_actions_p1, p=self.expert_policy_p1[state])
                action_p2_probs = np.ones(self.num_actions_p2) / self.num_actions_p2
                action_p2 = np.random.choice(self.num_actions_p2, p=action_p2_probs)
                next_state = np.random.choice(self.num_states, p=self.transition_P[state, action_p1, action_p2])
                
                # Check if expert player (Player 1) reached the reward cell
                next_state_coords = self.grid_game.map_state_idx_to_state(next_state)
                player1_pos = next_state_coords[0]  # Player 1's position
                
                if self.bc_cnn:
                    img_array = self.grid_game.render(state=state)
                    img_array = img_array.astype(np.float32) / 255.0  # Normalize pixel values
                    state_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                    if self.target_size is not None:
                        resize_transform = transforms.Resize(self.target_size, antialias=True)
                        state_tensor = resize_transform(state_tensor)
                    next_img_array = self.grid_game.render(state=next_state)
                    next_img_array = next_img_array.astype(np.float32) / 255.0
                    next_state_tensor = torch.from_numpy(next_img_array).permute(2, 0, 1)
                    if self.target_size is not None:
                        resize_transform = transforms.Resize(self.target_size, antialias=True)
                        next_state_tensor = resize_transform(next_state_tensor)
                    
                    self.trajectory_buffer_p2.push(
                        state_tensor,
                        action_p2,
                        action_p1,  # Expert action is from p1
                        next_state_tensor,
                    )
                else:
                    self.trajectory_buffer_p2.push(
                        state,
                        action_p2,
                        action_p1,  # Expert action is from p1
                        next_state,
                    )

                state = next_state

                # Stop only the current trajectory when the expert reaches its goal
                if player1_pos in self.grid_game.reward_coordinates:
                    logger.info(
                        f"Player 2 exploration, trajectory {k+1}: "
                        f"Expert Player 1 reached goal at {player1_pos}. "
                        f"Terminating current trajectory."
                    )
                    break

            # End the trajectory in the trajectory buffer
            self.trajectory_buffer_p2.end_trajectory()


            #     state = next_state
                
            # # End the trajectory in the trajectory buffer
            # self.trajectory_buffer_p2.end_trajectory()

            # if player1_pos in self.grid_game.reward_coordinates:
            #     break

            if max_transitions is not None and self.trajectory_buffer_p2.num_transitions() >= max_transitions:
                logger.info(f"Max transitions reached for Player 2: {self.trajectory_buffer_p2.num_transitions()} >= {max_transitions}")
                break

        logger.info(
            f"Player 2 exploration completed: "
            f"{self.trajectory_buffer_p2.num_trajectories()} trajectories, "
            f"{self.trajectory_buffer_p2.num_transitions()} transitions"
        )
        
        # extract data from replay buffer to train behavioral cloning for player 2
        logger.info("="*80)
        logger.info("PHASE 3: Behavioral Cloning")
        logger.info("="*80)
        
        bc_start_time = time.time()

        if self.bc_cnn:
            logger.info("Using CNN-based Behavioral Cloning")
 
            D_mu_nuE = []
            for transition in self.trajectory_buffer_p1.iterator():
                state = transition.state
                expert_action = transition.expert_action
                D_mu_nuE.append((state, expert_action))

            # extract data from replay buffer to train behavioral cloning for player 1
            D_nu_muE = []
            for transition in self.trajectory_buffer_p2.iterator():
                state = transition.state
                expert_action = transition.expert_action
                D_nu_muE.append((state, expert_action))
            
            num_traj_p1 = self.trajectory_buffer_p1.num_trajectories()
            num_traj_p2 = self.trajectory_buffer_p2.num_trajectories()
            logger.info(f"Total transitions collected: Player 1={len(D_nu_muE)}, Player 2={len(D_mu_nuE)}")
            logger.info(f"Total trajectories collected: Player 1={num_traj_p1}, Player 2={num_traj_p2}")
        
        else:
            logger.info("Using MLP-based Behavioral Cloning")
        
            # Extract all transitions from trajectory buffers
            D_mu_nuE = []
            for transition in self.trajectory_buffer_p1.iterator():
                state = transition.state_idx
                expert_action = transition.expert_action
                D_mu_nuE.append((state, expert_action))

            # extract data from replay buffer to train behavioral cloning for player 1
            D_nu_muE = []
            for transition in self.trajectory_buffer_p2.iterator():
                state = transition.state_idx
                expert_action = transition.expert_action
                D_nu_muE.append((state, expert_action))
            
            num_traj_p1 = self.trajectory_buffer_p1.num_trajectories()
            num_traj_p2 = self.trajectory_buffer_p2.num_trajectories()
            logger.info(f"Total transitions collected: Player 1={len(D_nu_muE)}, Player 2={len(D_mu_nuE)}")
            logger.info(f"Total trajectories collected: Player 1={num_traj_p1}, Player 2={num_traj_p2}")
        
        # If dataset_sizes is specified, train BC on multiple subsets
        if dataset_sizes is not None:
            policies_list = []
            losses_list = []
            trajectory_counts_list = []
            
            for size in dataset_sizes:
                logger.info(f"\n--- Training BC with dataset size: {size} trajectories ---")
                
                # Subset the data based on number of trajectories
                # Use the first 'size' trajectories from each player
                if self.bc_cnn:
                    logger.info("Using CNN-based Behavioral Cloning Subset")
                     # Player 1 subset
                    D_nu_muE_subset = []
                    trajectories_p1 = self.trajectory_buffer_p2.get_trajectories()
                    for i, trajectory in enumerate(trajectories_p1):
                        if i >= size:
                            break
                        for transition in trajectory:
                            state = transition.state
                            expert_action = transition.expert_action
                            D_nu_muE_subset.append((state, expert_action))
                    
                    # Player 2 subset
                    D_mu_nuE_subset = []
                    trajectories_p2 = self.trajectory_buffer_p1.get_trajectories()
                    for i, trajectory in enumerate(trajectories_p2):
                        if i >= size:
                            break
                        for transition in trajectory:
                            state = transition.state
                            expert_action = transition.expert_action
                            D_mu_nuE_subset.append((state, expert_action))

                else:
                    # Player 1 subset
                    D_nu_muE_subset = []
                    trajectories_p1 = self.trajectory_buffer_p2.get_trajectories()
                    for i, trajectory in enumerate(trajectories_p1):
                        if i >= size:
                            break
                        for transition in trajectory:
                            state = transition.state_idx
                            expert_action = transition.expert_action
                            D_nu_muE_subset.append((state, expert_action))
                    
                    # Player 2 subset
                    D_mu_nuE_subset = []
                    trajectories_p2 = self.trajectory_buffer_p1.get_trajectories()
                    for i, trajectory in enumerate(trajectories_p2):
                        if i >= size:
                            break
                        for transition in trajectory:
                            state = transition.state_idx
                            expert_action = transition.expert_action
                            D_mu_nuE_subset.append((state, expert_action))
                
                logger.info(f"Subset samples: Player 1={len(D_nu_muE_subset)}, Player 2={len(D_mu_nuE_subset)}")
                logger.info(f"Subset trajectories: {min(size, num_traj_p1)}, {min(size, num_traj_p2)}")
                
                # Train BC for Player 1
                learner_mu = BehavioralCloningSingleAgent(
                    num_states=self.num_states,
                    num_actions=self.num_actions_p1,
                    cnn_policy=self.bc_cnn
                )
                hat_mu, loss_mu = learner_mu.train(D_nu_muE_subset, epochs=epochs, device=self.device)
                logger.info(f"Player 1 BC training completed. Final loss: {loss_mu:.6f}")
                
                # Train BC for Player 2
                learner_nu = BehavioralCloningSingleAgent(
                    num_states=self.num_states,
                    num_actions=self.num_actions_p2,
                    cnn_policy=self.bc_cnn
                )
                hat_nu, loss_nu = learner_nu.train(D_mu_nuE_subset, epochs=epochs, device=self.device)
                logger.info(f"Player 2 BC training completed. Final loss: {loss_nu:.6f}")

                if self.bc_cnn:
                    policy_mu = np.zeros((self.num_states, self.num_actions_p1))
                    policy_nu = np.zeros((self.num_states, self.num_actions_p2))
                    for state_idx in range(self.num_states):
                        img_array = self.grid_game.render(state=state_idx)
                        img_array = img_array.astype(np.float32) / 255.0
                        state_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(self.device)
                        if self.target_size is not None:
                            resize_transform = transforms.Resize(self.target_size, antialias=True)
                            state_tensor = resize_transform(state_tensor)
                        state_tensor = state_tensor.unsqueeze(0)  # Add batch dim
                        with torch.no_grad():
                            action_probs_p1 = hat_mu.get_action_probs(state_tensor).cpu().numpy()
                            action_probs_p2 = hat_nu.get_action_probs(state_tensor).cpu().numpy()
                        policy_mu[state_idx] = action_probs_p1
                        policy_nu[state_idx] = action_probs_p2
                    hat_mu = policy_mu
                    hat_nu = policy_nu
                
                policies_list.append((hat_mu, hat_nu))
                losses_list.append((loss_mu, loss_nu))
                trajectory_counts_list.append((len(D_nu_muE_subset), len(D_mu_nuE_subset)))
            
            bc_total_time = time.time() - bc_start_time
            logger.info(f"\nTotal BC training time (all sizes): {format_time(bc_total_time)}")
            return policies_list, losses_list
        else:
            # Original behavior: train once on all data
            logger.info(f"Training BC for Player 1 with {len(D_nu_muE)} samples...")
            learner_mu = BehavioralCloningSingleAgent(
                num_states=self.num_states,
                num_actions=self.num_actions_p1,
                cnn_policy=self.bc_cnn
            )
            hat_mu, loss_mu = learner_mu.train(D_nu_muE, epochs=epochs, device=self.device)
            logger.info(f"Player 1 BC training completed. Final loss: {loss_mu:.6f}")

            logger.info(f"Training BC for Player 2 with {len(D_mu_nuE)} samples...")
            learner_nu = BehavioralCloningSingleAgent(
                num_states=self.num_states,
                num_actions=self.num_actions_p2,
                cnn_policy=self.bc_cnn
            )
            hat_nu, loss_nu = learner_nu.train(D_mu_nuE, epochs=epochs, device=self.device)
            logger.info(f"Player 2 BC training completed. Final loss: {loss_nu:.6f}")

            bc_total_time = time.time() - bc_start_time
            logger.info(f"Total BC training time: {format_time(bc_total_time)}")

            bc_total_time = time.time() - bc_start_time
            logger.info(f"Total BC training time: {format_time(bc_total_time)}")

            if self.bc_cnn:
                    policy_mu = np.zeros((self.num_states, self.num_actions_p1))
                    policy_nu = np.zeros((self.num_states, self.num_actions_p2))
                    for state_idx in range(self.num_states):
                        img_array = self.grid_game.render(state=state_idx)
                        img_array = img_array.astype(np.float32) / 255.0
                        state_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(self.device)
                        if self.target_size is not None:
                            resize_transform = transforms.Resize(self.target_size, antialias=True)
                            state_tensor = resize_transform(state_tensor)
                        state_tensor = state_tensor.unsqueeze(0)  # Add batch dim
                        with torch.no_grad():
                            action_probs_p1 = hat_mu.get_action_probs(state_tensor).cpu().numpy()
                            action_probs_p2 = hat_nu.get_action_probs(state_tensor).cpu().numpy()
                        policy_mu[state_idx] = action_probs_p1
                        policy_nu[state_idx] = action_probs_p2
                    hat_mu = policy_mu
                    hat_nu = policy_nu
            
            return (hat_mu, hat_nu), (loss_mu, loss_nu)

