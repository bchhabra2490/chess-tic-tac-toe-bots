import multiprocessing as mp
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .cttt_env import ChessTicTacToeEnv


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = torch.from_numpy(np.stack([b.state for b in batch])).float()
        actions = torch.tensor([b.action for b in batch], dtype=torch.long)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([b.next_state for b in batch])).float()
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def select_action(q_net: DQN, state: np.ndarray, legal_action_ids, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.choice(legal_action_ids)

    with torch.no_grad():
        s = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        q_values = q_net(s).cpu().numpy()[0]

    # Mask illegal actions by setting them very negative
    mask = np.full_like(q_values, -1e9)
    mask[legal_action_ids] = 0.0
    masked_q = q_values + mask
    return int(masked_q.argmax())


def train_dqn_selfplay(
    num_episodes: int = 10_000,
    batch_size: int = 128,
    gamma: float = 0.99,
    lr: float = 5e-4,  # Lowered from 1e-3 for stability
    target_update: int = 200,  # More frequent updates (was 500)
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 5_000,
    reward_scale: float = 0.1,  # Scale rewards to prevent large Q-values
):
    env = ChessTicTacToeEnv()
    state_dim = env.reset().shape[0]
    action_dim = env.num_actions

    policy_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer()

    step_count = 0
    last_loss = None

    def compute_epsilon(step: int) -> float:
        frac = max(0.0, min(1.0, 1.0 - step / float(epsilon_decay)))
        return epsilon_end + (epsilon_start - epsilon_end) * frac

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False

        while not done:
            legal_actions = env.get_legal_actions()
            legal_ids = [env._encode_action(a) for a in legal_actions]
            epsilon = compute_epsilon(step_count)
            action = select_action(policy_net, state, legal_ids, epsilon)

            next_state, reward, done, info = env.step(action)

            # Scale and clip rewards for stability
            reward = np.clip(reward * reward_scale, -1.0, 1.0)

            # In self-play we want the same network to learn from both sides.
            # Reward is from the perspective of the player who just moved.
            replay.push(state, action, reward, next_state, done)
            state = next_state
            step_count += 1

            # Gradient update
            if len(replay) >= batch_size:
                (
                    states_b,
                    actions_b,
                    rewards_b,
                    next_states_b,
                    dones_b,
                ) = replay.sample(batch_size)

                states_b = states_b.to(DEVICE)
                actions_b = actions_b.to(DEVICE)
                rewards_b = rewards_b.to(DEVICE)
                next_states_b = next_states_b.to(DEVICE)
                dones_b = dones_b.to(DEVICE)

                q_values = policy_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(next_states_b).max(1)[0]
                    target_q = rewards_b + gamma * (1.0 - dones_b) * next_q

                # Check for NaN/Inf values
                if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                    print(f"Warning: NaN/Inf in q_values at step {step_count}, skipping update")
                    continue
                if torch.isnan(target_q).any() or torch.isinf(target_q).any():
                    print(f"Warning: NaN/Inf in target_q at step {step_count}, skipping update")
                    continue

                loss = nn.MSELoss()(q_values, target_q)
                loss_value = loss.item()

                # Check for invalid loss (NaN/Inf only; allow large but finite losses)
                if np.isnan(loss_value) or np.isinf(loss_value):
                    print(f"Warning: Invalid loss {loss_value} at step {step_count}, skipping update")
                    continue

                last_loss = loss_value

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            # Periodically update target network
            if step_count % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 0:
            print(
                f"Episode {episode}/{num_episodes} - steps: {step_count}, "
                f"epsilon: {compute_epsilon(step_count):.3f}, "
                f"loss: {last_loss if last_loss is not None else 'N/A'}"
            )

    return policy_net


def worker_process(
    worker_id: int,
    num_episodes: int,
    state_dim: int,
    action_dim: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: int,
    reward_scale: float,
    transition_queue: mp.Queue,
    weight_queue: mp.Queue,
    step_counter: mp.Value,
):
    """Worker process that runs episodes and collects transitions."""
    env = ChessTicTacToeEnv()
    # Create a local network (will be updated from main process)
    local_net = DQN(state_dim, action_dim).to(DEVICE)
    local_net.eval()

    def compute_epsilon(step: int) -> float:
        frac = max(0.0, min(1.0, 1.0 - step / float(epsilon_decay)))
        return epsilon_end + (epsilon_start - epsilon_end) * frac

    def select_action_local(state: np.ndarray, legal_action_ids, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.choice(legal_action_ids)

        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            q_values = local_net(s).cpu().numpy()[0]

        mask = np.full_like(q_values, -1e9)
        mask[legal_action_ids] = 0.0
        masked_q = q_values + mask
        return int(masked_q.argmax())

    episode = 0
    while episode < num_episodes:
        # Try to get updated weights from main process (non-blocking)
        try:
            while True:
                weights = weight_queue.get_nowait()
                # Convert CPU tensors back to device
                weights_on_device = {k: v.to(DEVICE) for k, v in weights.items()}
                local_net.load_state_dict(weights_on_device)
        except:
            pass  # No new weights available

        state = env.reset()
        done = False

        while not done:
            # Get current step count for epsilon calculation
            with step_counter.get_lock():
                current_step = step_counter.value

            legal_actions = env.get_legal_actions()
            legal_ids = [env._encode_action(a) for a in legal_actions]
            epsilon = compute_epsilon(current_step)
            action = select_action_local(state, legal_ids, epsilon)

            next_state, reward, done, info = env.step(action)

            # Scale and clip rewards for stability
            reward = np.clip(reward * reward_scale, -1.0, 1.0)

            # Send transition to main process
            transition_queue.put((state.copy(), action, reward, next_state.copy(), done))

            state = next_state

            # Increment step counter
            with step_counter.get_lock():
                step_counter.value += 1

        episode += 1

    # Signal completion
    transition_queue.put(None)


def train_dqn_selfplay_parallel(
    num_episodes: int = 10_000,
    num_workers: int = None,
    batch_size: int = 128,
    gamma: float = 0.99,
    lr: float = 5e-4,  # Lowered from 1e-3 for stability
    target_update: int = 200,  # More frequent updates (was 500)
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 5_000,
    sync_frequency: int = 100,  # How often to sync weights to workers
    reward_scale: float = 0.1,  # Scale rewards to prevent large Q-values
):
    """
    Parallel version of DQN training using multiple worker processes.

    Args:
        num_workers: Number of parallel workers (defaults to CPU count)
        sync_frequency: How many steps between weight syncs to workers
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    env = ChessTicTacToeEnv()
    state_dim = env.reset().shape[0]
    action_dim = env.num_actions

    policy_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer()

    # Shared state
    transition_queue = mp.Queue(maxsize=10000)  # Buffer for transitions
    weight_queue = mp.Queue()  # For sending weights to workers
    step_counter = mp.Value("i", 0)  # Shared step counter

    # Start worker processes
    workers = []
    episodes_per_worker = num_episodes // num_workers
    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(
                i,
                episodes_per_worker,
                state_dim,
                action_dim,
                epsilon_start,
                epsilon_end,
                epsilon_decay,
                reward_scale,
                transition_queue,
                weight_queue,
                step_counter,
            ),
        )
        p.start()
        workers.append(p)

    print(f"Started {num_workers} worker processes")

    step_count = 0
    last_loss = None
    episodes_collected = 0
    completed_workers = 0

    # Initial weight sync (convert to CPU for serialization)
    initial_weights = {k: v.cpu().clone() for k, v in policy_net.state_dict().items()}
    for _ in range(num_workers):
        weight_queue.put(initial_weights)

    while completed_workers < num_workers:
        # Collect transitions from workers
        try:
            transition = transition_queue.get(timeout=1.0)
            if transition is None:
                completed_workers += 1
                continue

            state, action, reward, next_state, done = transition
            replay.push(state, action, reward, next_state, done)
            episodes_collected += done  # Count episodes when done=True
            step_count += 1

            # Training step
            if len(replay) >= batch_size:
                (
                    states_b,
                    actions_b,
                    rewards_b,
                    next_states_b,
                    dones_b,
                ) = replay.sample(batch_size)

                states_b = states_b.to(DEVICE)
                actions_b = actions_b.to(DEVICE)
                rewards_b = rewards_b.to(DEVICE)
                next_states_b = next_states_b.to(DEVICE)
                dones_b = dones_b.to(DEVICE)

                q_values = policy_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(next_states_b).max(1)[0]
                    target_q = rewards_b + gamma * (1.0 - dones_b) * next_q

                # Check for NaN/Inf values
                if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                    print(f"Warning: NaN/Inf in q_values at step {step_count}, skipping update")
                    continue
                if torch.isnan(target_q).any() or torch.isinf(target_q).any():
                    print(f"Warning: NaN/Inf in target_q at step {step_count}, skipping update")
                    continue

                loss = nn.MSELoss()(q_values, target_q)
                loss_value = loss.item()

                # Check for invalid loss (NaN/Inf only; allow large but finite losses)
                if np.isnan(loss_value) or np.isinf(loss_value):
                    print(f"Warning: Invalid loss {loss_value} at step {step_count}, skipping update")
                    continue

                last_loss = loss_value

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            # Periodically update target network and sync weights
            if step_count % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Sync weights to workers periodically
            if step_count % sync_frequency == 0:
                # Clear old weights from queue and send new ones
                try:
                    while not weight_queue.empty():
                        weight_queue.get_nowait()
                except:
                    pass
                # Send updated weights to all workers
                new_weights = {k: v.cpu().clone() for k, v in policy_net.state_dict().items()}
                for _ in range(num_workers):
                    try:
                        weight_queue.put(new_weights, block=False)
                    except:
                        pass  # Queue full, skip this sync

            # Progress logging
            if episodes_collected > 0 and episodes_collected % 100 == 0:
                eps = compute_epsilon(step_count, epsilon_start, epsilon_end, epsilon_decay)
                print(
                    f"Episodes collected: {episodes_collected}, steps: {step_count}, "
                    f"epsilon: {eps:.3f}, "
                    f"loss: {last_loss if last_loss is not None else 'N/A'}"
                )

        except:
            # Timeout or queue empty - continue
            pass

    # Cleanup workers
    for p in workers:
        p.join()

    print(f"Training complete. Total episodes: {episodes_collected}, steps: {step_count}")
    return policy_net


def compute_epsilon(step: int, epsilon_start: float, epsilon_end: float, epsilon_decay: int) -> float:
    """Helper function for epsilon calculation."""
    frac = max(0.0, min(1.0, 1.0 - step / float(epsilon_decay)))
    return epsilon_end + (epsilon_start - epsilon_end) * frac


def save_model(model: DQN, path: str = "cttt_dqn.pth"):
    torch.save(model.state_dict(), path)


def load_model(path: str, state_dim: int, action_dim: int) -> DQN:
    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


if __name__ == "__main__":
    import sys

    # Use parallel training by default, sequential if --sequential flag is passed
    use_parallel = "--sequential" not in sys.argv

    if use_parallel:
        print("Using parallel training with multiple workers")
        net = train_dqn_selfplay_parallel(num_episodes=10_000)
    else:
        print("Using sequential training")
        net = train_dqn_selfplay(num_episodes=10_000)
    save_model(net)
