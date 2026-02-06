"""
Shared Socket.IO client base class for Chess Tic-Tac-Toe bots.

All bots (MCTS, LLM, DQN, etc.) should subclass `BaseSocketBotClient`
and implement `_maybe_play_turn` (and optionally `_convert_cell_from_server`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import socketio

from rl.cttt_env import Action, ChessTicTacToeEnv, PLAYER_O, PLAYER_X


class BaseSocketBotClient:
    """
    Base Socket.IO client that:
      - connects to the server
      - finds or creates a room
      - keeps a local `ChessTicTacToeEnv` in sync with server gameState
      - calls `_maybe_play_turn` whenever it's our turn

    Subclasses must implement:
      - `_maybe_play_turn(self) -> None`

    Subclasses may override:
      - `_convert_cell_from_server` if they need a different pawn `dir` convention
    """

    def __init__(self, server_url: str):
        self.sio = socketio.Client()
        self.server_url = server_url

        self.room_id: Optional[str] = None
        self.player_id: Optional[str] = None  # "X" or "O"

        # Local env mirror of the server state
        self.env = ChessTicTacToeEnv()

        self._register_handlers()

    # ------------------------------------------------------------------
    # Socket.IO handlers (shared)
    # ------------------------------------------------------------------
    def _register_handlers(self) -> None:
        @self.sio.event
        def connect():
            print("[INFO] Connected; finding or creating room...")
            self.sio.emit("findOrCreateRoom")

        @self.sio.on("roomCreated")
        def on_room_created(data: Dict[str, Any]):
            self.room_id = data.get("roomId")
            self.player_id = data.get("playerId")
            print(f"[INFO] Room created: {self.room_id}, " f"we play as {self.player_id}. Waiting for opponent...")

        @self.sio.on("roomJoined")
        def on_room_joined(data: Dict[str, Any]):
            self.room_id = data.get("roomId")
            players = data.get("players") or []
            if self.player_id is None:
                for p in players:
                    if p.get("id") == self.sio.sid and p.get("playerId") in (PLAYER_X, PLAYER_O):
                        self.player_id = p["playerId"]
                        break
            # Fallback when sid matching fails (observed in practice):
            if self.player_id is None:
                self.player_id = PLAYER_O
            print(f"[INFO] Joined room: {self.room_id}, we play as {self.player_id}.")

            game_state = data.get("gameState")
            if game_state:
                self._sync_env_from_game_state(game_state)
                self._maybe_play_turn()

        @self.sio.on("gameStateUpdate")
        def on_game_state_update(game_state: Dict[str, Any]):
            self._sync_env_from_game_state(game_state)

            if game_state.get("gameOver"):
                winner = game_state.get("winner")
                if winner is None:
                    print("[INFO] Game over: draw.")
                elif winner == self.player_id:
                    print("[INFO] Game over: we WIN.")
                else:
                    print("[INFO] Game over: we LOSE.")
                self.sio.disconnect()
                return

            self._maybe_play_turn()

        @self.sio.on("onlineCount")
        def on_online_count(count: int):
            print(f"[INFO] Online players: {count}")

        @self.sio.on("playerDisconnected")
        def on_player_disconnected(data: Dict[str, Any]):
            print(f"[INFO] Player disconnected: {data}")

        @self.sio.on("error")
        def on_error(data: Dict[str, Any]):
            print(f"[ERROR] Server: {data}")

        @self.sio.event
        def disconnect():
            print("[INFO] Disconnected.")

    # ------------------------------------------------------------------
    # Environment sync helpers
    # ------------------------------------------------------------------
    def _convert_cell_from_server(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a JS-side board cell dict to the Python env representation.

        Default implementation passes `dir` through unchanged; subclasses can
        override if they need to invert or otherwise transform it.
        """
        return {
            "player": cell.get("player"),
            "type": cell.get("type"),
            "dir": cell.get("dir"),
        }

    def _sync_env_from_game_state(self, game_state: Dict[str, Any]) -> None:
        """Mirror the server's gameState into our Python env."""
        # Board
        board = []
        for cell in game_state.get("board", []):
            if cell is None:
                board.append(None)
            else:
                board.append(self._convert_cell_from_server(cell))
        self.env.board = board

        # Pools
        pools = game_state.get("pools", {})
        self.env.pools = {
            PLAYER_X: list(pools.get("X", [])),
            PLAYER_O: list(pools.get("O", [])),
        }

        # Turn / status
        self.env.current_player = game_state.get("currentPlayer", PLAYER_X)
        self.env.game_over = bool(game_state.get("gameOver", False))
        self.env.winner = game_state.get("winner")
        # `move_count` isn't sent by the server; we don't rely on it.
        self.env.move_count = 0

    # ------------------------------------------------------------------
    # Move sending helper
    # ------------------------------------------------------------------
    def _emit_move(self, action: Action) -> None:
        """Send an `Action` to the server as a Socket.IO `makeMove` payload."""
        if action.kind == "place":
            payload = {
                "action": "place",
                "index": action.place_index,
                "type": action.piece_type,
            }
        elif action.kind == "move":
            payload = {
                "action": "move",
                "fromIndex": action.from_index,
                "toIndex": action.to_index,
            }
        else:
            print(f"[ERROR] Unknown action kind: {action.kind}")
            return

        print(f"[INFO] Playing ({self.player_id}): {payload}")
        self.sio.emit("makeMove", payload)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _maybe_play_turn(self) -> None:  # pragma: no cover - abstract
        """Called after each gameStateUpdate when it's potentially our turn."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        print(f"[INFO] Connecting to {self.server_url} ...")
        self.sio.connect(self.server_url)
        self.sio.wait()
