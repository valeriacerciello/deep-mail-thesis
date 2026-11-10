import numpy as np
import logging
from pathlib import Path
from datetime import datetime



# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Use free GPUs

import torch
from collections import namedtuple

import os

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Define the structure for storing transitions in the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'expert_action', 'next_state'))

def setup_logging(log_file='deep_uniform_training.log'):
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
    logger = logging.getLogger('DeepUniform')
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

class ActionRewardValueIteration:
    """
    A Value Iteration solver for MDPs where rewards depend on state-action pairs R(s,a).
    """
    def __init__(self, epsilon, max_iter):
        self.eps = epsilon
        self.max_iter = max_iter

    def run_algo(self, P, R_sa, gamma=0.99):
        """
        Args:
            P: Transition model, shape (S, A, S).
            R_sa: Reward matrix, shape (S, A), where R_sa[s,a] is reward for taking action a in state s.
            gamma: Discount factor.

        Returns:
            V: Optimal state-value function, shape (S,).
            pi: Optimal deterministic policy, one-hot encoded, shape (S, A).
        """
        S, A, _ = P.shape
        V = np.zeros(S)

        for it in range(self.max_iter):
            # Compute Q(s,a) = R_sa[s,a] + γ ∑_{s'} P[s,a,s'] * V[s']
            # This is the key change: we use R_sa directly instead of a state-only R_s.
            Q = R_sa + gamma * np.einsum('sat,t->sa', P, V)

            V_new = np.max(Q, axis=1)
            if np.max(np.abs(V_new - V)) < self.eps:
                V = V_new
                break
            V = V_new

        # Extract final Q-table for accurate policy extraction
        Q_final = R_sa + gamma * np.einsum('sat,t->sa', P, V)
        
        best_actions = np.argmax(Q_final, axis=1)
        pi = np.zeros((S, A), dtype=float)
        pi[np.arange(S), best_actions] = 1.0
        
        # We also return the final Q-table, which is needed for finding ties
        return V, pi, Q_final
    

def policy_value_zero_sum( mu_pi: np.ndarray, nu_pi: np.ndarray, transition: np.ndarray, reward: np.ndarray, initial_dist: np.ndarray, gamma: float) -> float:
    """
    Evaluate the *joint* value V^{mu, \nu} = E[ ∑ gamma^t r(s_t,a_t,b_t) ] 
    by solving the linear system (I - gammaP_π)^-1 r_π and averaging over
    initial uniform state.
    """
    # Build 1-step expected reward per state
    r_s = (reward * mu_pi[:, :, None] * nu_pi[:, None, :]).sum(axis=(1,2))
    # Build joint transition matrix P_π[s,s'] = ∑_{a,b} μ(a|s) ν(b|s) P[s,a,b,s']
    P_joint = (transition * mu_pi[:, :, None, None] * nu_pi[:, None, :, None]).sum(axis=(1,2))
    identity_matrix = np.eye(transition.shape[0])  # Avoid ambiguous variable name
    V = np.linalg.solve(identity_matrix - gamma * P_joint, r_s)

    return float(initial_dist @ V)

def value_iteration( R: np.ndarray, P: np.ndarray, initial_dist: np.ndarray,
                        tol: float = 1e-6, max_iter: int = 10_000, gamma:float=0.9) -> float:
    """
    Standard value iteration for single-agent MDP:
        R: shape (S, A)
        P: shape (S, A, S')
    Returns optimal state-value averaged under uniform start.
    """
    S, A = R.shape
    V = np.zeros(S)
    for _ in range(max_iter):
        # Q(s,a) = R[s,a] + gamma ∑ P[s,a,s'] V[s']
        Q = R + gamma * (P @ V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return float(initial_dist @ V)

def calc_exploitability_true_both( mu_pi: np.ndarray, nu_pi: np.ndarray, reward: np.ndarray, transition: np.ndarray, initial_dist: np.ndarray, gamma: float) -> float:
        """
        Compute exact exploitability of joint policy under true rewards.
        
        Args:
            mu_pi (np.ndarray): Player μ policy (maximizer), shape (S, A1)
            nu_pi (np.ndarray): Player ν policy (minimizer), shape (S, A2)
            
        Returns:
            float: Exploitability value. Lower values indicate better policies.
                  Zero exploitability corresponds to Nash equilibrium.
        """
        # 1) Compute V^{μ,ν} by solving the full zero‐sum game via simple policy eval
        V_joint = policy_value_zero_sum(mu_pi, nu_pi, transition=transition, reward=reward, initial_dist=initial_dist, gamma=gamma)

        # 2) Build single‐agent MDP for μ as decision‐maker:
        #    R_μ(s,a) = E_{b∼ν_pi(s)}[ r(s,a,b) ]
        R_mu = (reward * nu_pi[:, None, :]).sum(axis=2)  # shape (S, A1)
         #    P_mu[s,a,s'] = ∑_b ν_pi(b|s) P[s,a,b,s']
        P_mu = (transition * nu_pi[:, None, :, None]).sum(axis=2)  # shape (S, A1, S')
        # best‐response value for μ:
        V_br_mu = value_iteration(R_mu, P_mu, initial_dist=initial_dist, gamma=gamma)

        # 3) ν's induced MDP (with negated rewards):
        #    R_nu[s,b] = - E_{a∼μ_pi(s)}[ r(s,a,b) ]
        R_nu = - (reward * mu_pi[:, :, None]).sum(axis=1)  # (S, A2)
        #    P_nu[s,b,s'] = ∑_a μ_pi(a|s) P[s,a,b,s']
        P_nu = (transition * mu_pi[:, :, None, None]).sum(axis=1)  # (S, A2, S')
        V_br_nu = value_iteration(R_nu, P_nu, initial_dist=initial_dist, gamma=gamma)

        return float(max(V_br_mu - V_joint, V_br_nu + V_joint))

def calc_exploitability_true_mu( mu_pi: np.ndarray, nu_pi: np.ndarray, reward: np.ndarray, transition: np.ndarray, initial_dist: np.ndarray, gamma: float) -> float:
        """
        Compute exact exploitability of joint policy under true rewards.
        
        Args:
            mu_pi (np.ndarray): Player μ policy (maximizer), shape (S, A1)
            nu_pi (np.ndarray): Player ν policy (minimizer), shape (S, A2)
            
        Returns:
            float: Exploitability value. Lower values indicate better policies.
                  Zero exploitability corresponds to Nash equilibrium.
        """
        # 1) Compute V^{μ,ν} by solving the full zero‐sum game via simple policy eval
        V_joint = policy_value_zero_sum(mu_pi, nu_pi, transition=transition, reward=reward, initial_dist=initial_dist, gamma=gamma)

        # 2) Build single‐agent MDP for μ as decision‐maker:
        #    R_μ(s,a) = E_{b∼ν_pi(s)}[ r(s,a,b) ]
        R_mu = (reward * nu_pi[:, None, :]).sum(axis=2)  # shape (S, A1)
         #    P_mu[s,a,s'] = ∑_b ν_pi(b|s) P[s,a,b,s']
        P_mu = (transition * nu_pi[:, None, :, None]).sum(axis=2)  # shape (S, A1, S')
        # best‐response value for μ:
        V_br_mu = value_iteration(R_mu, P_mu, initial_dist=initial_dist, gamma=gamma)

        return float(V_br_mu - V_joint)

def calc_exploitability_true_nu( mu_pi: np.ndarray, nu_pi: np.ndarray, reward: np.ndarray, transition: np.ndarray, initial_dist: np.ndarray, gamma: float) -> float:
        """
        Compute exact exploitability of joint policy under true rewards.
        
        Args:
            mu_pi (np.ndarray): Player μ policy (maximizer), shape (S, A1)
            nu_pi (np.ndarray): Player ν policy (minimizer), shape (S, A2)
            
        Returns:
            float: Exploitability value. Lower values indicate better policies.
                  Zero exploitability corresponds to Nash equilibrium.
        """
        # 1) Compute V^{μ,ν} by solving the full zero‐sum game via simple policy eval
        V_joint = policy_value_zero_sum(mu_pi, nu_pi, transition=transition, reward=reward, initial_dist=initial_dist, gamma=gamma)

        # 3) ν's induced MDP (with negated rewards):
        #    R_nu[s,b] = - E_{a∼μ_pi(s)}[ r(s,a,b) ]
        R_nu = - (reward * mu_pi[:, :, None]).sum(axis=1)  # (S, A2)
        #    P_nu[s,b,s'] = ∑_a μ_pi(a|s) P[s,a,b,s']
        P_nu = (transition * mu_pi[:, :, None, None]).sum(axis=1)  # (S, A2, S')
        V_br_nu = value_iteration(R_nu, P_nu, initial_dist=initial_dist, gamma=gamma)

        return float(V_br_nu - V_joint)


def setup_experiment_logging(log_file='deepmail_vs_deepunif.log'):
    """Set up logging for the comparison experiment."""
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'{timestamp}_{log_file}'
    
    logger = logging.getLogger('DeepMAIL_vs_DeepUnif')
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