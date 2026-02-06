"""
Socket.IO client that uses a trained DQN to play Chess Tic-Tac-Toe
against a human (or another client) on the existing Node.js server.

Usage:
  1. Start the Node server:
       npm start
  2. Start a browser client in Online mode (or another bot instance).
  3. In this folder (project root or rl/), run:
       python -m rl.dqn_socket_client
"""

import sys
from typing import Any, Dict

import numpy as np
import torch

from .cttt_env import ChessTicTacToeEnv, Action, PLAYER_X, PLAYER_O
from .dqn_selfplay import DQN, load_model, select_action, DEVICE
from socket_bot_base import BaseSocketBotClient


SERVER_URL = "http://localhost:3000"
MODEL_PATH = "cttt_dqn.pth"


class DQNBotClient(BaseSocketBotClient):
    def __init__(self, server_url: str = SERVER_URL, model_path: str = MODEL_PATH):
        super().__init__(server_url=server_url)

        # Local env is created by the base class; we just need its shapes.
        dummy_state = self.env.reset()
        state_dim = dummy_state.shape[0]
        action_dim = self.env.num_actions

        # Load trained DQN
        try:
            self.policy_net: DQN = load_model(model_path, state_dim, action_dim)
        except FileNotFoundError:
            print(f"[ERROR] Model file '{model_path}' not found. Train first via rl/dqn_selfplay.py.")
            sys.exit(1)

    def _maybe_play_turn(self) -> None:
        """If it's our turn, use the DQN to pick and send a move."""
        if self.player_id is None:
            return
        if self.env.game_over:
            return
        if self.env.current_player != self.player_id:
            # Not our turn
            return

        # Encode current state
        state = self.env._encode_state()
        legal_actions = self.env.get_legal_actions()
        if not legal_actions:
            print("[WARN] No legal actions available.")
            return

        legal_ids = [self.env._encode_action(a) for a in legal_actions]

        # Greedy action selection (epsilon = 0)
        action_id = select_action(self.policy_net, state, legal_ids, epsilon=0.0)
        action_map = {self.env._encode_action(a): a for a in legal_actions}
        chosen: Action = action_map[action_id]

        print(f"[INFO] Our turn ({self.player_id}).")
        self._emit_move(chosen)


def main():
    client = DQNBotClient()
    client.run()


if __name__ == "__main__":
    main()
