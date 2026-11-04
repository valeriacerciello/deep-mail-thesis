# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Use free GPUs

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
import copy
from torchvision import transforms
from tqdm import tqdm
import logging
from datetime import datetime
import time
import os

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from additional_experiments.bc import BehavioralCloningSingleAgent


# Define the structure for storing transitions in the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'expert_action', 'reward', 'next_state', 'state_idx'))


def setup_logging(log_file='deep_mail_training.log'):
    """
    Set up logging configuration for training.
    
    Args:
        log_file: Name of the log file to write to
    
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'{timestamp}_{log_file}'
    
    # Configure logger
    logger = logging.getLogger('DeepMAIL')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_path}")
    
    return logger, log_path


def format_time(seconds):
    """
    Format seconds into a human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


class ReplayBuffer:
    """A simple replay buffer."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))
        if len(self.memory) > self.memory.maxlen:
            self.memory.popleft()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()

    def iterator(self):
        return iter(self.memory)

    def __len__(self):
        return len(self.memory)


class TrajectoryBuffer:
    """
    A replay buffer without capacity limit that stores all trajectories collected 
    during the exploration phase (before running BC).
    """
    def __init__(self, player_id):
        """
        Initialize the trajectory buffer.
        
        Args:
            player_id: Integer identifier for the player (e.g., 1 or 2)
        """
        self.player_id = player_id
        self.trajectories = []  # List of trajectories, where each trajectory is a list of transitions
        self.current_trajectory = []  # Current trajectory being built
        
    def push(self, *args):
        """
        Add a transition to the current trajectory.
        
        Args:
            *args: Transition components (state, action, expert_action, reward, next_state, state_idx)
        """
        self.current_trajectory.append(Transition(*args))
    
    def end_trajectory(self):
        """
        Mark the current trajectory as complete and start a new one.
        """
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = []
    
    def get_all_transitions(self):
        """
        Get all transitions from all trajectories as a flat list.
        
        Returns:
            List of all Transition objects
        """
        all_transitions = []
        for trajectory in self.trajectories:
            all_transitions.extend(trajectory)
        # Include current trajectory if it's not empty
        if self.current_trajectory:
            all_transitions.extend(self.current_trajectory)
        return all_transitions
    
    def get_trajectories(self):
        """
        Get all complete trajectories.
        
        Returns:
            List of trajectories (list of lists of Transitions)
        """
        return self.trajectories
    
    def num_trajectories(self):
        """Return the number of complete trajectories."""
        return len(self.trajectories)
    
    def num_transitions(self):
        """Return the total number of transitions across all trajectories."""
        return sum(len(traj) for traj in self.trajectories) + len(self.current_trajectory)
    
    def save(self, filepath):
        """
        Save the trajectory buffer to a file.
        
        Args:
            filepath: Path where the buffer should be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Convert trajectories to a serializable format
        data = {
            'player_id': self.player_id,
            'num_trajectories': self.num_trajectories(),
            'num_transitions': self.num_transitions(),
            'trajectories': []
        }
        
        for trajectory in self.trajectories:
            traj_data = []
            for transition in trajectory:
                # Convert tensors to numpy for serialization
                traj_data.append({
                    'state': transition.state.cpu().numpy() if isinstance(transition.state, torch.Tensor) else transition.state,
                    'action': transition.action.cpu().numpy() if isinstance(transition.action, torch.Tensor) else transition.action,
                    'expert_action': transition.expert_action.cpu().numpy() if isinstance(transition.expert_action, torch.Tensor) else transition.expert_action,
                    'reward': transition.reward.cpu().numpy() if isinstance(transition.reward, torch.Tensor) else transition.reward,
                    'next_state': transition.next_state.cpu().numpy() if isinstance(transition.next_state, torch.Tensor) else transition.next_state,
                    'state_idx': transition.state_idx
                })
            data['trajectories'].append(traj_data)
        
        # Save to file using numpy
        np.save(filepath, data, allow_pickle=True)
        print(f"Saved trajectory buffer for Player {self.player_id} to {filepath}")
        print(f"  - {data['num_trajectories']} trajectories")
        print(f"  - {data['num_transitions']} total transitions")
    
    def load(self, filepath, device='cpu'):
        """
        Load the trajectory buffer from a file.
        
        Args:
            filepath: Path to the saved buffer file
            device: Torch device to load tensors to
        """
        data = np.load(filepath, allow_pickle=True).item()
        
        self.player_id = data['player_id']
        self.trajectories = []
        
        for traj_data in data['trajectories']:
            trajectory = []
            for trans_data in traj_data:
                # Convert numpy arrays back to tensors
                transition = Transition(
                    state=torch.from_numpy(trans_data['state']).to(device),
                    action=torch.from_numpy(trans_data['action']).to(device) if isinstance(trans_data['action'], np.ndarray) else torch.tensor(trans_data['action'], device=device),
                    expert_action=torch.from_numpy(trans_data['expert_action']).to(device) if isinstance(trans_data['expert_action'], np.ndarray) else torch.tensor(trans_data['expert_action'], device=device),
                    reward=torch.from_numpy(trans_data['reward']).to(device) if isinstance(trans_data['reward'], np.ndarray) else torch.tensor(trans_data['reward'], device=device),
                    next_state=torch.from_numpy(trans_data['next_state']).to(device),
                    state_idx=trans_data['state_idx']
                )
                trajectory.append(transition)
            self.trajectories.append(trajectory)
        
        print(f"Loaded trajectory buffer for Player {self.player_id} from {filepath}")
        print(f"  - {self.num_trajectories()} trajectories")
        print(f"  - {self.num_transitions()} total transitions")
    
    def clear(self):
        """Clear all trajectories."""
        self.trajectories = []
        self.current_trajectory = []
    
    def __len__(self):
        """Return the total number of transitions."""
        return self.num_transitions()
    
    def iterator(self):
        """Iterator over all transitions in the buffer."""
        return iter(self.get_all_transitions())

class DQN(nn.Module):
    """Modified DQN"""
    def __init__(self, image_size, in_channels, n_actions, hidden_dim=512, tau=0.005):
        super(DQN, self).__init__()
        # for now we use a cnn with two hidden layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        conved_size = image_size - 3*(3 - 1)  # after three conv layers with kernel size 3
        self.fc1 = nn.Linear(128 * conved_size * conved_size, hidden_dim)
        # after the first linear layer we concatenate the action
        self.fc2 = nn.Linear(hidden_dim + n_actions, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # output Q-value$
        self.tau = tau  # for soft update of target network
        self.n_actions = n_actions

    def _forward_until_action(self, state):
        x = self.conv1(state)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        return x

    def _get_inner_layers_embedding(self, state, action):
        x = self._forward_until_action(state)
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        return x

    def forward(self, state, action):
        x = self._get_inner_layers_embedding(state, action)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
    def get_all_q_values(self, state):
        """Vectorized method to get Q-values for all actions for a given state."""
        batch_size = state.size(0)
        # expand state to evaluate all actions in parallel
        state_expanded = state.repeat_interleave(self.n_actions, dim=0)
        # create a batch of all possible one-hot encoded actions
        all_actions = torch.eye(self.n_actions, device=state.device).repeat(batch_size, 1)
        # get all Q-values in a single forward pass
        q_values = self(state_expanded, all_actions)
        # reshape to (batch_size, n_actions)
        return q_values.view(batch_size, self.n_actions)

    def train_on_replay_buffer(self, replay_buffer, target_network, epochs=1, batch_size=64, gamma=0.9, lr=1e-3, device='cuda'):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for epoch in range(epochs):
            if len(replay_buffer) < batch_size:
                continue
            transitions = replay_buffer.sample(batch_size)
            batch = Transition(*zip(*transitions))

            state_batch = torch.stack(batch.state)
            action_batch = F.one_hot(torch.stack(batch.action), num_classes=self.n_actions).float()
            reward_batch = torch.stack(batch.reward).unsqueeze(1)
            next_state_batch = torch.stack(batch.next_state)

            # Compute Q(s_t, a)
            q_values = self(state_batch, action_batch)

            # Vectorized computation of V(s_{t+1}) using target network
            with torch.no_grad():
                # Get Q-values for all possible next actions in one pass
                next_q_values_all_actions = target_network.get_all_q_values(next_state_batch)
                # V(s_{t+1}) = max_a' Q_target(s_{t+1}, a')
                next_q_values, _ = torch.max(next_q_values_all_actions, dim=1, keepdim=True)

            # Compute the target Q values
            target_q_values = reward_batch + (gamma * next_q_values)

            # Compute loss
            loss = loss_fn(q_values, target_q_values)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Soft update of the target network should be done after all epochs for this training session
            for target_param, local_param in zip(target_network.parameters(), self.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

class DeepWARMMAIL:
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
        image_size,
        in_channels,
        grid_game,
        dqn_hidden_dim=512,
        replay_buffer_capacity=1000,
        batch_size=64,
        gamma=0.9,
        lr=1e-3,
        tau=0.005,
        beta=10.0, # regularization parameter
        temperature=1.0, # exploration parameter,
        bc_cnn=False,
        device='cpu',
        save_buffers=False
    ):
        self.K = K
        self.num_states = num_states
        self.num_actions_p1 = num_actions_p1
        self.num_actions_p2 = num_actions_p2
        self.expert_policy_p1 = expert_policy_p1
        self.expert_policy_p2 = expert_policy_p2
        self.transition_P = transition_P
        self.initial_state_sampler = initial_state_sampler
        self.image_size = image_size
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.beta = beta
        self.temperature = temperature
        self.device = device
        self.feature_dim = dqn_hidden_dim
        self.grid_game = grid_game
        self.target_size = (image_size, image_size)  # Target size for image resizing
        self.bc_cnn = bc_cnn
        self.save_buffers = save_buffers
        # Initialize DQNs for both players
        self.dqn_p1 = DQN(image_size, in_channels, num_actions_p1, dqn_hidden_dim).to(device)
        # the target networks are copies of the original DQNs and are used for stable training
        self.dqn_target_p1 = copy.deepcopy(self.dqn_p1).to(device)
        self.dqn_optimizer_p1 = optim.Adam(self.dqn_p1.parameters(), lr=lr)

        self.dqn_p2 = DQN(image_size, in_channels, num_actions_p2, dqn_hidden_dim).to(device)
        self.dqn_target_p2 = copy.deepcopy(self.dqn_p2).to(device)
        self.dqn_optimizer_p2 = optim.Adam(self.dqn_p2.parameters(), lr=lr)

        # Replay buffers for both players (with capacity for training)
        self.replay_buffer_p1 = ReplayBuffer(replay_buffer_capacity)
        self.replay_buffer_p2 = ReplayBuffer(replay_buffer_capacity)
        
        # Trajectory buffers for both players (without capacity, stores all exploration data)
        self.trajectory_buffer_p1 = TrajectoryBuffer(player_id=1)
        self.trajectory_buffer_p2 = TrajectoryBuffer(player_id=2)

    def _get_reward(self, lambda_matrix_inv, dqn):
        """Compute the reward function using the pre-computed inverse of lambda."""
        def reward_function(state, action):
            with torch.no_grad():
                inner_layers = dqn._get_inner_layers_embedding(state, action)
                # Use torch.einsum for a batched dot product, which is more robust
                reward = torch.sqrt(torch.einsum('bi,ij,bj->b', inner_layers, lambda_matrix_inv, inner_layers))
            return reward
        return reward_function

    def run(self, horizon, lambda_reg=0.1, epochs=1000, gradient_steps=20,t_max=1000, logger=None, dataset_sizes=None):
        """Main training loop for Deep WARM-MAIL.
        
        Args:
            horizon: Episode horizon
            lambda_reg: Regularization parameter (unused)
            epochs: Number of BC training epochs
            gradient_steps: Gradient steps per Q learning update 
            t_max: Max iterations per WARM-MAIL iteration
            logger: Logger instance
            dataset_sizes: List of dataset sizes to train BC on (e.g., [1, 50, 100, 500]).
                          If None, trains once on all collected data.
        
        Returns:
            If dataset_sizes is None:
                ((hat_mu, hat_nu), (loss_mu, loss_nu), (num_traj_mu, num_traj_nu))
            If dataset_sizes is provided:
                (policies_list, losses_list, trajectory_counts_list)
                where each element corresponds to a dataset size
        """
        if logger is None:
            logger = logging.getLogger('DeepMAIL')
        
        # Start overall timing
        overall_start_time = time.time()
        
        logger.info("="*80)
        logger.info("Starting Deep WARM-MAIL Training")
        logger.info("="*80)
        logger.info("Configuration:")
        logger.info(f"  K (WARM-MAIL iterations): {self.K}")
        logger.info(f"  Horizon: {horizon}")
        logger.info(f"  BC Epochs: {epochs}")
        logger.info(f"  t_max per iteration: {t_max}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.lr}")
        logger.info(f"  Gamma (discount): {self.gamma}")
        logger.info(f"  Beta (exploration): {self.beta}")
        logger.info(f"  Temperature (exploration): {self.temperature}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Image size: {self.image_size}")
        logger.info(f"  Feature dim: {self.feature_dim}")
        if dataset_sizes is not None:
            logger.info(f"  Dataset sizes for BC: {dataset_sizes}")
        logger.info("")
        
        # run exploration for the first player
        logger.info("="*80)
        logger.info("PHASE 1: Training Player 1")
        logger.info("="*80)
        # initialize lambda lambda to identity matrix
        lambda_matrix_p1 = torch.eye(self.feature_dim).to(self.device) * self.beta
        
        # Track time for Player 1 phase
        p1_start_time = time.time()
        p1_iteration_times = []
        
        for k in tqdm(range(self.K)):
            iteration_start_time = time.time()
            
            logger.info(f"\n--- Player 1: WARM-MAIL Iteration {k+1}/{self.K} ---")
            old_dqn_p1 = copy.deepcopy(self.dqn_p1) # Keep a frozen copy for feature extraction
            lambda_inv_p1 = torch.linalg.inv(lambda_matrix_p1)
            reward_function_p1 = self._get_reward(lambda_inv_p1, old_dqn_p1)
            lambda_new_p1 = lambda_matrix_p1.clone()
            lambda_logdet_p1 = torch.logdet(lambda_matrix_p1).item()
            logger.info(f"Initial lambda log-determinant: {lambda_logdet_p1:.4f}")
            pbar = tqdm(total=t_max, desc=f"P1 Epoch {k+1}/{self.K}", leave=False)
            iter_count = 0
            trajectories_collected = 0
            while iter_count < t_max and torch.logdet(lambda_new_p1).item() <= lambda_logdet_p1 + np.log(2):
                # sample the trajectory
                state = self.initial_state_sampler()
                new_trajectory_features = []
                trajectories_collected += 1
                for _ in range(horizon):
                    # get q value and use softmax policy to sample action for player 1
                    img_array = self.grid_game.render(state=state)
                    # Convert to float32 and normalize to [0, 1]
                    img_array = img_array.astype(np.float32) / 255.0
                    # Convert to tensor and permute to [C, H, W] format for PyTorch
                    state_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(self.device)
                    
                    if self.target_size is not None:
                        # Resize expects [C, H, W] format
                        resize_transform = transforms.Resize(self.target_size, antialias=True)
                        state_tensor = resize_transform(state_tensor)
                    
                    # Add batch dimension for DQN: [C, H, W] -> [1, C, H, W]
                    state_tensor = state_tensor.unsqueeze(0)

                    with torch.no_grad():
                        q_values = self.dqn_p1.get_all_q_values(state_tensor).squeeze(0)
                        
                        # Check for NaN or Inf in Q-values
                        if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                            print(f"Warning: NaN or Inf detected in Q-values at iteration {k+1}/{self.K}, trajectory {trajectories_collected}")
                            print(f"Q-values: {q_values}")
                            # Use uniform distribution as fallback
                            action_p1_probs = np.ones(self.num_actions_p1) / self.num_actions_p1
                        else:
                            # Clip Q-values to prevent overflow in softmax
                            q_values = torch.clamp(q_values / self.temperature, min=-50, max=50)
                            action_p1_probs = F.softmax(q_values, dim=0).cpu().numpy()
                            
                            # Additional safety check
                            if np.isnan(action_p1_probs).any():
                                print(f"Warning: NaN in softmax output at iteration {k+1}/{self.K}, trajectory {trajectories_collected}")
                                action_p1_probs = np.ones(self.num_actions_p1) / self.num_actions_p1
                                
                    action_p1 = np.random.choice(self.num_actions_p1, p=action_p1_probs)
                    action_p2 = np.random.choice(self.num_actions_p2, p=self.expert_policy_p2[state])

                    next_state = np.random.choice(self.num_states, p=self.transition_P[state, action_p1, action_p2])
                    
                    # Check if expert player (Player 2) reached the reward cell if so, restart the trajectory
                    next_state_coords = self.grid_game.map_state_idx_to_state(next_state)
                    player2_pos = next_state_coords[1]  # Player 2's position
                    

                    # compute reward (explorative reward)
                    action_tensor = F.one_hot(torch.tensor([action_p1]), num_classes=self.num_actions_p1).float().to(self.device)
                    reward = reward_function_p1(state_tensor, action_tensor).item()
                    with torch.no_grad():
                        phi = old_dqn_p1._get_inner_layers_embedding(state_tensor, action_tensor)
                        new_trajectory_features.append(phi)

                    # Prepare transition data (reuse state_tensor, but remove batch dim for storage)
                    state_preprocessed = state_tensor.squeeze(0)  # [1, C, H, W] -> [C, H, W]
                    action_p1_tensor = torch.tensor(action_p1, dtype=torch.long, device=self.device)
                    action_p2_tensor = torch.tensor(action_p2, dtype=torch.long, device=self.device)
                    reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
                    img_array = self.grid_game.render(state=next_state)
                    # Convert to float32 and normalize to [0, 1]
                    img_array = img_array.astype(np.float32) / 255.0
                    # Convert to tensor and permute to [C, H, W] format for PyTorch
                    next_state_preprocessed = torch.from_numpy(img_array).permute(2, 0, 1).to(self.device)
                    if self.target_size is not None:
                        resize_transform = transforms.Resize(self.target_size, antialias=True)
                        next_state_preprocessed = resize_transform(next_state_preprocessed)
                    
                    # Store transition in replay buffer (for DQN training)
                    self.replay_buffer_p1.push(
                        state_preprocessed,
                        action_p1_tensor,
                        action_p2_tensor,
                        reward_tensor,
                        next_state_preprocessed,
                        state  # Store the original state index
                    )
                    
                    # Store transition in trajectory buffer (for BC training later)
                    self.trajectory_buffer_p1.push(
                        state_preprocessed,
                        action_p1_tensor,
                        action_p2_tensor,
                        reward_tensor,
                        next_state_preprocessed,
                        state  # Store the original state index
                    )
                    
                    state = next_state
                    if player2_pos in self.grid_game.reward_coordinates:
                        break
                
                # End the trajectory in the trajectory buffer
                self.trajectory_buffer_p1.end_trajectory()
                
                # update DQN for player 1 by training on the whole replay buffer
                self.dqn_p1.train_on_replay_buffer(self.replay_buffer_p1, self.dqn_target_p1, epochs=gradient_steps, batch_size=self.batch_size, device=self.device)
                
                # update lambda matrix
                if new_trajectory_features:
                    phi_batch = torch.cat(new_trajectory_features, dim=0)
                    lambda_new_p1 += torch.matmul(phi_batch.T, phi_batch)

                iter_count += 1
                pbar.update(1)
                current_logdet = torch.logdet(lambda_new_p1).item()
                pbar.set_postfix({'logdet_diff': f'{current_logdet - lambda_logdet_p1:.2f}'})
            pbar.close()
            
            final_logdet_p1 = torch.logdet(lambda_new_p1).item()
            logger.info(f"Trajectories collected: {trajectories_collected}")
            logger.info(f"Final lambda log-determinant: {final_logdet_p1:.4f}")
            logger.info(f"Log-determinant increase: {final_logdet_p1 - lambda_logdet_p1:.4f}")
            logger.info(f"Replay buffer size: {len(self.replay_buffer_p1)}")
            
            # Track iteration time and estimate remaining time
            iteration_time = time.time() - iteration_start_time
            p1_iteration_times.append(iteration_time)
            logger.info(f"Iteration time: {format_time(iteration_time)}")
            
            if len(p1_iteration_times) > 0:
                avg_iteration_time = np.mean(p1_iteration_times)
                remaining_iterations = self.K - (k + 1)
                estimated_remaining = avg_iteration_time * remaining_iterations
                if remaining_iterations > 0:
                    logger.info(f"Estimated time remaining (Player 1): {format_time(estimated_remaining)}")

            # update lambda matrix for next iteration $\Lambda = \sum_{x,a \in \mathcal{D}_n} \mathrm{OldInitialLayers}(x,a) \mathrm{OldInitialLayers}(x,a)^\trans + \beta I$
            with torch.no_grad():
                # Note: This can be memory intensive if the replay buffer is huge.
                # Consider sampling if memory becomes an issue.
                all_transitions = self.replay_buffer_p1.memory
                if not all_transitions: continue
                
                buffer_batch = Transition(*zip(*all_transitions))
                state_tensors = torch.stack(buffer_batch.state)
                action_tensors = F.one_hot(torch.stack(buffer_batch.action), num_classes=self.num_actions_p1).float()
                
                phi_vectors = self.dqn_p1._get_inner_layers_embedding(state_tensors, action_tensors)
                lambda_matrix_p1 = torch.matmul(phi_vectors.T, phi_vectors) + torch.eye(self.feature_dim, device=self.device) * self.beta
            
        p1_total_time = time.time() - p1_start_time
        logger.info("\nPlayer 1 training completed!")
        logger.info(f"Final replay buffer size: {len(self.replay_buffer_p1)}")
        logger.info(f"Total Player 1 training time: {format_time(p1_total_time)}")
        logger.info("")
        
        # run exploration for the second player
        logger.info("="*80)
        logger.info("PHASE 2: Training Player 2")
        logger.info("="*80)
        lambda_matrix_p2 = torch.eye(self.feature_dim, device=self.device) * self.beta
        
        # Track time for Player 2 phase
        p2_start_time = time.time()
        p2_iteration_times = []
        
        for k in tqdm(range(self.K), desc="Player 2 Training"):
            iteration_start_time = time.time()
            
            logger.info(f"\n--- Player 2: WARM-MAIL Iteration {k+1}/{self.K} ---")
            # 1. Pre-compute matrix inverse (major optimization)
            old_dqn_p2 = copy.deepcopy(self.dqn_p2) # Keep a frozen copy for feature extraction
            lambda_inv_p2 = torch.linalg.inv(lambda_matrix_p2)
            reward_function_p2 = self._get_reward(lambda_inv_p2, old_dqn_p2)

            lambda_new_p2 = lambda_matrix_p2.clone()
            lambda_logdet_p2 = torch.logdet(lambda_matrix_p2).item()
            logger.info(f"Initial lambda log-determinant: {lambda_logdet_p2:.4f}")

            pbar = tqdm(total=t_max, desc=f"P2 Epoch {k+1}/{self.K}", leave=False)
            iter_count = 0
            trajectories_collected = 0
            while iter_count < t_max and torch.logdet(lambda_new_p2).item() <= lambda_logdet_p2 + np.log(2):
                state = self.initial_state_sampler()
                new_trajectory_features = []
                trajectories_collected += 1

                for _ in range(horizon):
                    # Player 1 uses the expert policy this time
                    action_p1 = np.random.choice(self.num_actions_p1, p=self.expert_policy_p1[state])
                    
                    # 2. Vectorized action selection for Player 2 (major optimization)
                    img_array = self.grid_game.render(state=state)
                    # Convert to float32 and normalize to [0, 1]
                    img_array = img_array.astype(np.float32) / 255.0
                    # Convert to tensor and permute to [C, H, W] format for PyTorch
                    state_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(self.device)
                    if self.target_size is not None:
                        resize_transform = transforms.Resize(self.target_size, antialias=True)
                        state_tensor = resize_transform(state_tensor)
                    
                    # Add batch dimension for DQN: [C, H, W] -> [1, C, H, W]
                    state_tensor = state_tensor.unsqueeze(0)
                    
                    with torch.no_grad():
                        q_values = self.dqn_p2.get_all_q_values(state_tensor).squeeze(0)
                        
                        # Check for NaN or Inf in Q-values
                        if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                            print(f"Warning: NaN or Inf detected in Player 2 Q-values at iteration {k+1}/{self.K}, trajectory {trajectories_collected}")
                            print(f"Q-values: {q_values}")
                            # Use uniform distribution as fallback
                            action_p2_probs = np.ones(self.num_actions_p2) / self.num_actions_p2
                        else:
                            # Clip Q-values to prevent overflow in softmax
                            q_values = torch.clamp(q_values / self.temperature, min=-50, max=50)
                            action_p2_probs = F.softmax(q_values, dim=0).cpu().numpy()
                            
                            # Additional safety check
                            if np.isnan(action_p2_probs).any():
                                print(f"Warning: NaN in Player 2 softmax output at iteration {k+1}/{self.K}, trajectory {trajectories_collected}")
                                action_p2_probs = np.ones(self.num_actions_p2) / self.num_actions_p2
                                
                    action_p2 = np.random.choice(self.num_actions_p2, p=action_p2_probs)

                    next_state = np.random.choice(self.num_states, p=self.transition_P[state, action_p1, action_p2])
                    
                    # Check if expert player (Player 1) reached the reward cell
                    next_state_coords = self.grid_game.map_state_idx_to_state(next_state)
                    player1_pos = next_state_coords[0]  # Player 1's position

                    # Compute reward and features
                    action_tensor = F.one_hot(torch.tensor([action_p2]), num_classes=self.num_actions_p2).float().to(self.device)
                    reward = reward_function_p2(state_tensor, action_tensor).item()
                    with torch.no_grad():
                        phi = old_dqn_p2._get_inner_layers_embedding(state_tensor, action_tensor)
                        new_trajectory_features.append(phi)
                    
                    # Prepare transition data (reuse state_tensor, but remove batch dim for storage)
                    state_preprocessed = state_tensor.squeeze(0)  # [1, C, H, W] -> [C, H, W]
                    action_p2_tensor = torch.tensor(action_p2, dtype=torch.long, device=self.device)
                    action_p1_tensor = torch.tensor(action_p1, dtype=torch.long, device=self.device)
                    reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
                    img_array = self.grid_game.render(state=next_state)
                    # Convert to float32 and normalize to [0, 1]
                    img_array = img_array.astype(np.float32) / 255.0
                    # Convert to tensor and permute to [C, H, W] format for PyTorch
                    next_state_preprocessed = torch.from_numpy(img_array).permute(2, 0, 1).to(self.device)
                    if self.target_size is not None:
                        resize_transform = transforms.Resize(self.target_size, antialias=True)
                        next_state_preprocessed = resize_transform(next_state_preprocessed)
                    
                    # Store transition in Player 2's replay buffer (for DQN training)
                    self.replay_buffer_p2.push(
                        state_preprocessed,
                        action_p2_tensor,
                        action_p1_tensor,  # Expert action is from p1
                        reward_tensor,
                        next_state_preprocessed,
                        state  # Store the original state index
                    )
                    
                    # Store transition in trajectory buffer (for BC training later)
                    self.trajectory_buffer_p2.push(
                        state_preprocessed,
                        action_p2_tensor,
                        action_p1_tensor,  # Expert action is from p1
                        reward_tensor,
                        next_state_preprocessed,
                        state  # Store the original state index
                    )
                    
                    state = next_state
                    if player1_pos in self.grid_game.reward_coordinates:
                        break
                
                # End the trajectory in the trajectory buffer
                self.trajectory_buffer_p2.end_trajectory()
                
                # Update DQN for player 2
                self.dqn_p2.train_on_replay_buffer(
                    self.replay_buffer_p2,
                    self.dqn_target_p2,
                    epochs=gradient_steps,
                    batch_size=self.batch_size,
                    device=self.device
                )
                
                
                # Update lambda matrix with features from the new trajectory
                if new_trajectory_features:
                    phi_batch = torch.cat(new_trajectory_features, dim=0)
                    lambda_new_p2 += torch.matmul(phi_batch.T, phi_batch)

                iter_count += 1
                pbar.update(1)
                current_logdet = torch.logdet(lambda_new_p2).item()
                pbar.set_postfix({'logdet_diff': f'{current_logdet - lambda_logdet_p2:.2f}'})
            pbar.close()
            
            final_logdet_p2 = torch.logdet(lambda_new_p2).item()
            logger.info(f"Trajectories collected: {trajectories_collected}")
            logger.info(f"Final lambda log-determinant: {final_logdet_p2:.4f}")
            logger.info(f"Log-determinant increase: {final_logdet_p2 - lambda_logdet_p2:.4f}")
            logger.info(f"Replay buffer size: {len(self.replay_buffer_p2)}")
            
            # Track iteration time and estimate remaining time
            iteration_time = time.time() - iteration_start_time
            p2_iteration_times.append(iteration_time)
            logger.info(f"Iteration time: {format_time(iteration_time)}")
            
            if len(p2_iteration_times) > 0:
                avg_iteration_time = np.mean(p2_iteration_times)
                remaining_iterations = self.K - (k + 1)
                estimated_remaining = avg_iteration_time * remaining_iterations
                if remaining_iterations > 0:
                    logger.info(f"Estimated time remaining (Player 2): {format_time(estimated_remaining)}")

            # Rebuild lambda from the entire replay buffer using the updated network
            with torch.no_grad():
                all_transitions = self.replay_buffer_p2.memory
                if not all_transitions: continue
                
                buffer_batch = Transition(*zip(*all_transitions))
                state_tensors = torch.stack(buffer_batch.state)
                action_tensors = F.one_hot(torch.stack(buffer_batch.action), num_classes=self.num_actions_p2).float()
                
                phi_vectors = self.dqn_p2._get_inner_layers_embedding(state_tensors, action_tensors)
                lambda_matrix_p2 = torch.matmul(phi_vectors.T, phi_vectors) + torch.eye(self.feature_dim, device=self.device) * self.beta
    
        p2_total_time = time.time() - p2_start_time
        logger.info("\nPlayer 2 training completed!")
        logger.info(f"Final replay buffer size: {len(self.replay_buffer_p2)}")
        logger.info(f"Total Player 2 training time: {format_time(p2_total_time)}")
        logger.info("")
        
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

            bc_total_time = time.time() - bc_start_time
            logger.info(f"Total BC training time: {format_time(bc_total_time)}")
        
        # Save trajectory buffers to files if flag is enabled
        if self.save_buffers:
            logger.info("\n" + "="*80)
            logger.info("Saving Trajectory Buffers")
            logger.info("="*80)
            
            # Create data directory if it doesn't exist
            data_dir = Path(__file__).parent.parent / 'data' / 'trajectories'
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for the filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save Player 1 trajectory buffer
            p1_buffer_path = data_dir / f'deep_mail_bufferp1_K{self.K}_{timestamp}.npy'
            self.trajectory_buffer_p1.save(str(p1_buffer_path))
            logger.info(f"Player 1 trajectory buffer saved to: {p1_buffer_path}")
            
            # Save Player 2 trajectory buffer
            p2_buffer_path = data_dir / f'deep_mail_bufferp2_K{self.K}_{timestamp}.npy'
            self.trajectory_buffer_p2.save(str(p2_buffer_path))
            logger.info(f"Player 2 trajectory buffer saved to: {p2_buffer_path}")
            logger.info("")
        
        # Calculate overall time
        overall_total_time = time.time() - overall_start_time
        
        logger.info("\n" + "="*80)
        logger.info("Deep WARM-MAIL Training Completed Successfully!")
        logger.info("="*80)
        logger.info(f"Training Time Summary:")
        logger.info(f"  Player 1 training: {format_time(p1_total_time)} ({p1_total_time/overall_total_time*100:.1f}%)")
        logger.info(f"  Player 2 training: {format_time(p2_total_time)} ({p2_total_time/overall_total_time*100:.1f}%)")
        logger.info(f"  BC training: {format_time(bc_total_time)} ({bc_total_time/overall_total_time*100:.1f}%)")
        logger.info(f"  Total time: {format_time(overall_total_time)}")
        logger.info("="*80 + "\n")
        
        # Return based on whether dataset_sizes was specified
        if dataset_sizes is not None:
            return policies_list, losses_list, trajectory_counts_list
        else:
            # Original return format: also return the number of trajectories used to train each player
            return (hat_mu, hat_nu), (loss_mu, loss_nu), (len(D_nu_muE), len(D_mu_nuE))
