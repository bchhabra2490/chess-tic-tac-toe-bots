"""
Minimax bot that plays Chess Tic-Tac-Toe as an online player.

Uses a simple depth-limited minimax (with alpha-beta pruning) over the
Python `ChessTicTacToeEnv`, connected to the Node.js server via Socket.IO.
"""

import copy
import math
import os
import random
import sys
from typing import Dict, Optional, Tuple

# Allow importing from bots/rl (same repo)
_BOTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _BOTS_DIR not in sys.path:
    sys.path.insert(0, _BOTS_DIR)

from cttt_env import Action, ChessTicTacToeEnv, PLAYER_O, PLAYER_X, BOARD_SIZE
from socket_bot_base import BaseSocketBotClient


SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:3000")
MINIMAX_MAX_DEPTH = int(os.environ.get("MINIMAX_MAX_DEPTH", "3"))


def clone_env(env: ChessTicTacToeEnv) -> ChessTicTacToeEnv:
    """Create a deep copy of the environment for search."""
    new_env = ChessTicTacToeEnv()
    new_env.board = [copy.deepcopy(cell) if cell is not None else None for cell in env.board]
    new_env.pools = {
        PLAYER_X: env.pools[PLAYER_X].copy(),
        PLAYER_O: env.pools[PLAYER_O].copy(),
    }
    new_env.current_player = env.current_player
    new_env.game_over = env.game_over
    new_env.winner = env.winner
    new_env.move_count = env.move_count
    return new_env


def evaluate_position(env: ChessTicTacToeEnv, root_player: str) -> float:
    """
    Heuristic evaluation of a non-terminal position from root_player's perspective.

    Very simple:
      - +1 for a win, -1 for a loss, 0 for draw.
      - Otherwise, difference in pieces on board scaled down.
    """
    if env.winner == root_player:
        return 1.0
    if env.winner is not None and env.winner != root_player:
        return -1.0
    if env.game_over:
        return 0.0

    # Material heuristic: more pieces on board is (slightly) better.
    def count_pieces(player: str) -> int:
        return sum(1 for cell in env.board if cell is not None and cell["player"] == player)

    my_pieces = count_pieces(root_player)
    opp = PLAYER_O if root_player == PLAYER_X else PLAYER_X
    opp_pieces = count_pieces(opp)

    return 0.1 * (my_pieces - opp_pieces)


def minimax_search(env: ChessTicTacToeEnv, root_player: str, max_depth: int) -> Action:
    """
    Depth-limited minimax with alpha-beta pruning.

    Returns the best action for `root_player` in the given env.
    """
    legal_actions = env.get_legal_actions()
    if not legal_actions:
        raise ValueError("No legal actions available")

    def is_root_turn(e: ChessTicTacToeEnv) -> bool:
        return e.current_player == root_player

    def search(e: ChessTicTacToeEnv, depth: int, alpha: float, beta: float) -> float:
        # Terminal or depth limit
        if e.game_over or depth == 0:
            return evaluate_position(e, root_player)

        legal = e.get_legal_actions()
        if not legal:
            # No legal moves: treat as draw-ish
            return 0.0

        if is_root_turn(e):
            value = -math.inf
            for a in legal:
                child = clone_env(e)
                action_id = child._encode_action(a)
                child.step(action_id)
                value = max(value, search(child, depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for a in legal:
                child = clone_env(e)
                action_id = child._encode_action(a)
                child.step(action_id)
                value = min(value, search(child, depth - 1, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # Top-level: choose action with best value
    best_value = -math.inf
    best_actions = []

    for a in legal_actions:
        child = clone_env(env)
        action_id = child._encode_action(a)
        child.step(action_id)
        val = search(child, max_depth - 1, -math.inf, math.inf)
        if val > best_value:
            best_value = val
            best_actions = [a]
        elif val == best_value:
            best_actions.append(a)

    # Break ties randomly to add variety
    return random.choice(best_actions) if best_actions else random.choice(legal_actions)


class MinimaxBotClient(BaseSocketBotClient):
    def __init__(self, server_url: str = SERVER_URL):
        super().__init__(server_url=server_url)

    def _convert_cell_from_server(self, cell: Dict) -> Dict:
        """
        Match the convention used by the Python env for pawn `dir`.

        JS uses `forwardRow = fr - dir`, Python env uses `forward_row = fr + dir`,
        so we invert `dir` when syncing from the server.
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
        if self.player_id is None or self.env.game_over or self.env.current_player != self.player_id:
            return

        legal_actions = self.env.get_legal_actions()
        if not legal_actions:
            print("[WARN] No legal actions for minimax bot.")
            return

        print(f"[INFO] Minimax thinking... (depth: {MINIMAX_MAX_DEPTH})")
        try:
            chosen = minimax_search(self.env, self.player_id, MINIMAX_MAX_DEPTH)
        except Exception as e:
            print(f"[WARN] Minimax failed: {e}, choosing random move")
            chosen = random.choice(legal_actions)

        self._emit_move(chosen)

    def run(self) -> None:
        print(f"[INFO] Minimax config: depth={MINIMAX_MAX_DEPTH}")
        super().run()


def main():
    client = MinimaxBotClient(server_url=SERVER_URL)
    client.run()


if __name__ == "__main__":
    main()
