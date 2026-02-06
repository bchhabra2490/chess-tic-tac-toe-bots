"""
LLM bot that plays Chess Tic-Tac-Toe as an online player via Socket.IO.

Uses an LLM (OpenAI API by default) to choose moves. Requires OPENAI_API_KEY.

Usage:
  1. Start the Node server from chess-tic-tac-toe/:  npm start
  2. Run this bot from project root:
       python bots/llm-bot.py
     Or with custom server URL:
       SERVER_URL=http://localhost:3000 python bots/llm-bot.py
  3. Open the game in a browser in Online mode; the bot will join and play.
"""

import os
import random
import re
import sys
from typing import Any, Dict, List, Optional

# Allow importing from bots/rl (same repo)
_BOTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _BOTS_DIR not in sys.path:
    sys.path.insert(0, _BOTS_DIR)

from cttt_env import Action, ChessTicTacToeEnv, PLAYER_O, PLAYER_X, BOARD_SIZE, BOARD_CELLS, PIECE_TYPES
from socket_bot_base import BaseSocketBotClient

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:3000")


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible)
# ---------------------------------------------------------------------------


def get_llm_move(
    board_text: str,
    our_player: str,
    legal_moves_text: str,
    api_key: Optional[str] = None,
    model: str = "gpt-5",
) -> Optional[str]:
    """Call LLM to choose one move. Returns the raw response line or None."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[WARN] OPENAI_API_KEY not set; will pick a random legal move.")
        return None

    try:
        import openai
    except ImportError:
        print("[WARN] openai package not installed; pip install openai. Using random move.")
        return None

    client = openai.OpenAI(api_key=api_key)
    system = """You are playing Chess Tic-Tac-Toe on a 4x4 board.

Rules:
- Each player has one of each: Pawn (P), Rook (R), Knight (N), Bishop (B).
- First phase: take turns placing pieces on empty squares until each side has 3 pieces on the board.
- After that, on your turn you may either place your remaining piece on an empty square, or move one of your pieces using standard chess moves (rook: same row/column; bishop: diagonal; knight: L-shape; pawn: one square in its current direction, or diagonal to capture).
- Pawns do not promote; when a pawn reaches any board edge (row 0 or row 3), on future moves it moves in the opposite direction.
- Captured pieces go back to their owner's pool and can be placed again on a later turn.
- Rows and columns are 0–3. Row 0 is the top, row 3 is the bottom.
- First to get 4 of their pieces in a row (horizontal, vertical, or diagonal) wins.

You must reply with exactly one move in one of these formats:
- PLACE <row> <col> <piece>   (e.g. PLACE 0 1 R)
- MOVE <from_row> <from_col> <to_row> <to_col>   (e.g. MOVE 1 0 2 0)
Use only the row and column numbers and piece letters from the board description. Reply with nothing else."""

    user = f"""Current game state: Board (row, col 0-3; empty = .):
{board_text}

You play as {our_player}. Your legal moves:
{legal_moves_text}

Reply with exactly one move: PLACE row col piece OR MOVE from_row from_col to_row to_col."""

    try:
        print(f"[INFO] Calling LLM with model: {model}")
        resp = client.chat.completions.create(
            model=model,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        print(f"[INFO] LLM response: {resp.choices[0].message.content}")
        raw = (resp.choices[0].message.content or "").strip()
        return raw if raw else None
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}")
        return None


def parse_llm_response(
    raw: Optional[str],
    legal_actions: List[Action],
    env: ChessTicTacToeEnv,
) -> Optional[Action]:
    """Parse LLM response into an Action. Returns None if parsing fails."""
    if not raw:
        return None
    raw_upper = raw.upper().strip()
    # PLACE row col piece
    m = re.match(r"PLACE\s+(\d)\s+(\d)\s+([PRNB])", raw_upper, re.I)
    if m:
        row, col, piece = int(m.group(1)), int(m.group(2)), m.group(3).upper()
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            idx = row * BOARD_SIZE + col
            for a in legal_actions:
                if a.kind == "place" and a.place_index == idx and a.piece_type == piece:
                    return a
    # MOVE from_row from_col to_row to_col
    m = re.match(r"MOVE\s+(\d)\s+(\d)\s+(\d)\s+(\d)", raw_upper)
    if m:
        r1, c1, r2, c2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if all(0 <= x < BOARD_SIZE for x in (r1, c1, r2, c2)):
            from_idx = r1 * BOARD_SIZE + c1
            to_idx = r2 * BOARD_SIZE + c2
            for a in legal_actions:
                if a.kind == "move" and a.from_index == from_idx and a.to_index == to_idx:
                    return a
    return None


def format_board(env: ChessTicTacToeEnv) -> str:
    """Human-readable 4x4 board with row/col labels."""
    lines = ["  0 1 2 3"]
    for r in range(BOARD_SIZE):
        row_str = f"{r}"
        for c in range(BOARD_SIZE):
            idx = r * BOARD_SIZE + c
            cell = env.board[idx]
            if cell is None:
                row_str += " ."
            else:
                row_str += f" {cell['player']}{cell['type']}"
        lines.append(row_str)
    return "\n".join(lines)


def format_legal_moves_with_env(legal_actions: List[Action], env: ChessTicTacToeEnv) -> str:
    lines = []
    for a in legal_actions:
        if a.kind == "place":
            row, col = env._index_to_row_col(a.place_index)
            lines.append(f"  PLACE {row} {col} {a.piece_type}")
        else:
            r1, c1 = env._index_to_row_col(a.from_index)
            r2, c2 = env._index_to_row_col(a.to_index)
            lines.append(f"  MOVE {r1} {c1} {r2} {c2}")
    return "\n".join(lines) if lines else "  (none)"


class LLMBotClient(BaseSocketBotClient):
    def __init__(self, server_url: str = SERVER_URL):
        super().__init__(server_url=server_url)

    def _maybe_play_turn(self) -> None:
        if self.player_id is None or self.env.game_over or self.env.current_player != self.player_id:
            return

        legal_actions = self.env.get_legal_actions()
        if not legal_actions:
            print("[WARN] No legal actions.")
            return

        board_text = format_board(self.env)
        legal_moves_text = format_legal_moves_with_env(legal_actions, self.env)
        raw = get_llm_move(board_text, self.player_id, legal_moves_text)
        chosen = parse_llm_response(raw, legal_actions, self.env)
        if chosen is None:
            chosen = random.choice(legal_actions)
            print("[INFO] LLM parse failed or no API key; chose random move.")

        self._emit_move(chosen)

    def run(self) -> None:
        super().run()


def main():
    client = LLMBotClient(server_url=SERVER_URL)
    client.run()


if __name__ == "__main__":
    main()
