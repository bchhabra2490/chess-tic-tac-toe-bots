"""
Monte Carlo Tree Search (MCTS) bot for Chess Tic-Tac-Toe.

Uses MCTS algorithm to choose moves. Connects via Socket.IO as an online player.

Usage:
  1. Start the Node server from chess-tic-tac-toe/:  npm start
  2. Run this bot from project root:
       python bots/mcts-bot.py
     Or with custom server URL:
       SERVER_URL=http://localhost:3000 python bots/mcts-bot.py
  3. Open the game in a browser in Online mode; the bot will join and play.
"""

import copy
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Allow importing from bots/rl (same repo)
_BOTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _BOTS_DIR not in sys.path:
    sys.path.insert(0, _BOTS_DIR)

from rl.cttt_env import Action, ChessTicTacToeEnv, PLAYER_O, PLAYER_X, BOARD_SIZE
from socket_bot_base import BaseSocketBotClient

SERVER_URL = os.environ.get("SERVER_URL", "https://chess-tic-tac-toe-production.up.railway.app")
MCTS_ITERATIONS = int(os.environ.get("MCTS_ITERATIONS", "1000"))  # Number of MCTS simulations
MCTS_TIME_LIMIT = float(os.environ.get("MCTS_TIME_LIMIT", "2.0"))  # Max seconds per move


# ---------------------------------------------------------------------------
# MCTS Implementation
# ---------------------------------------------------------------------------


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(
        self,
        action: Optional[Action] = None,
        parent: Optional["MCTSNode"] = None,
        player: str = PLAYER_X,
    ):
        self.action = action  # Action that led to this node
        self.parent = parent
        self.player = player  # Player whose turn it is at this node
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.wins = 0.0  # Total reward from this node's perspective
        self.untried_actions: List[Action] = []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0 and len(self.children) > 0

    def is_terminal(self, env: ChessTicTacToeEnv) -> bool:
        return env.game_over

    def ucb1_value(self, exploration: float = math.sqrt(2)) -> float:
        """UCB1 value for selection."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration_term

    def select_child(self) -> "MCTSNode":
        """Select child with highest UCB1 value."""
        return max(self.children, key=lambda c: c.ucb1_value())

    def add_child(self, action: Action, player: str) -> "MCTSNode":
        """Add a new child node."""
        child = MCTSNode(action=action, parent=self, player=player)
        self.children.append(child)
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        return child

    def update(self, reward: float) -> None:
        """Backpropagate reward."""
        self.visits += 1
        self.wins += reward


def clone_env(env: ChessTicTacToeEnv) -> ChessTicTacToeEnv:
    """Create a deep copy of the environment for simulation."""
    new_env = ChessTicTacToeEnv()
    # Deep copy board
    new_env.board = [copy.deepcopy(cell) if cell is not None else None for cell in env.board]
    # Deep copy pools
    new_env.pools = {
        PLAYER_X: env.pools[PLAYER_X].copy(),
        PLAYER_O: env.pools[PLAYER_O].copy(),
    }
    new_env.current_player = env.current_player
    new_env.game_over = env.game_over
    new_env.winner = env.winner
    new_env.move_count = env.move_count
    return new_env


def random_rollout(env: ChessTicTacToeEnv, root_player: str) -> float:
    """
    Simulate a random game from current state.
    Returns reward from root_player's perspective: +1 win, -1 loss, 0 draw.
    """
    sim_env = clone_env(env)
    max_moves = 100  # Safety limit
    moves = 0

    while not sim_env.game_over and moves < max_moves:
        legal_actions = sim_env.get_legal_actions()
        if not legal_actions:
            break

        # Random action
        action = random.choice(legal_actions)
        action_id = sim_env._encode_action(action)
        _, _, done, _ = sim_env.step(action_id)
        moves += 1

        if done:
            break

    # Determine reward from root_player's perspective
    if sim_env.winner == root_player:
        return 1.0
    elif sim_env.winner is not None:
        return -1.0
    else:
        return 0.0  # Draw


def mcts_search(env: ChessTicTacToeEnv, root_player: str, iterations: int, time_limit: float) -> Action:
    """
    Perform MCTS search and return best action.

    Args:
        env: Current game state
        root_player: Player we're choosing a move for
        iterations: Max number of MCTS iterations
        time_limit: Max time in seconds

    Returns:
        Best action according to MCTS
    """
    root = MCTSNode(player=root_player)

    # Initialize untried actions
    root.untried_actions = env.get_legal_actions()

    if not root.untried_actions:
        raise ValueError("No legal actions available")

    start_time = time.time()
    iteration = 0

    while iteration < iterations and (time.time() - start_time) < time_limit:
        # Selection: traverse to leaf
        node = root
        sim_env = clone_env(env)

        # Traverse tree
        while node.is_fully_expanded() and not node.is_terminal(sim_env):
            node = node.select_child()
            # Apply action to simulation env
            if node.action:
                action_id = sim_env._encode_action(node.action)
                sim_env.step(action_id)

        # If this node has never had its actions initialized, do it now
        # so that deeper parts of the tree can expand as well.
        if not node.untried_actions and not node.is_terminal(sim_env):
            node.untried_actions = sim_env.get_legal_actions()

        # Expansion: add one child if not terminal
        if not node.is_terminal(sim_env) and node.untried_actions:
            action = random.choice(node.untried_actions)
            action_id = sim_env._encode_action(action)
            sim_env.step(action_id)

            # After step, env.current_player is now the opponent
            # So the child node represents the opponent's turn
            next_player = sim_env.current_player
            child = node.add_child(action, next_player)
            node = child

        # Simulation: random rollout
        reward = random_rollout(sim_env, root_player)

        # Backpropagation: update all nodes from leaf to root.
        # Reward is always from the root player's perspective, so we do
        # NOT flip the sign as we traverse up the tree.
        while node is not None:
            node.update(reward)
            node = node.parent

        iteration += 1

    # Choose best action (most visited child)
    if not root.children:
        # Fallback: pick random untried action
        return random.choice(root.untried_actions)

    best_child = max(root.children, key=lambda c: c.visits)
    print(
        f"[INFO] MCTS: {iteration} iterations, best action visits: {best_child.visits}, win rate: {best_child.wins/best_child.visits:.2%}"
    )

    return best_child.action


class MCTSBotClient(BaseSocketBotClient):
    def __init__(self, server_url: str = SERVER_URL):
        super().__init__(server_url=server_url)

    def _convert_cell_from_server(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JS-side pawn `dir` convention to the Python env's convention.

        JS uses `forwardRow = fr - dir`, while the env uses `forward_row = fr + dir`,
        so we invert the sign of `dir` when syncing from the server.
        """
        dir_val = cell.get("dir")
        if isinstance(dir_val, int):
            dir_val = -dir_val
        return {
            "player": cell.get("player"),
            "type": cell.get("type"),
            "dir": dir_val,
        }

    def _maybe_play_turn(self) -> None:
        """If it's our turn, use MCTS to choose and send a move."""
        if self.player_id is None or self.env.game_over or self.env.current_player != self.player_id:
            return

        legal_actions = self.env.get_legal_actions()
        if not legal_actions:
            print("[WARN] No legal actions.")
            return

        # Use MCTS to choose best move
        print(f"[INFO] Thinking... (MCTS iterations: {MCTS_ITERATIONS}, time limit: {MCTS_TIME_LIMIT}s)")
        try:
            chosen = mcts_search(
                self.env,
                self.player_id,
                iterations=MCTS_ITERATIONS,
                time_limit=MCTS_TIME_LIMIT,
            )
        except Exception as e:
            print(f"[WARN] MCTS failed: {e}, choosing random move")
            chosen = random.choice(legal_actions)

        self._emit_move(chosen)

    def run(self) -> None:
        print(f"[INFO] MCTS config: {MCTS_ITERATIONS} iterations, {MCTS_TIME_LIMIT}s time limit")
        super().run()


def main():
    client = MCTSBotClient(server_url=SERVER_URL)
    client.run()


if __name__ == "__main__":
    main()
