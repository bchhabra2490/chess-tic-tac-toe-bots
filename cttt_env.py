import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


PLAYER_X = "X"
PLAYER_O = "O"
BOARD_SIZE = 4
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
PIECE_TYPES = ["P", "R", "N", "B"]


@dataclass
class Action:
    """Unified action representation used internally by the env."""

    kind: str  # "place" or "move"
    from_index: Optional[int] = None
    to_index: Optional[int] = None
    place_index: Optional[int] = None
    piece_type: Optional[str] = None


class ChessTicTacToeEnv:
    """
    Self-play environment for the Chess Tic-Tac-Toe variant.

    This is a faithful port of the game logic from `game.js` but implemented
    in Python and exposing a gym-like API:
      - reset() -> obs
      - step(action_id) -> (obs, reward, done, info)

    The action space is a fixed discrete set of size 320:
      - 0..255  : moves  (from_index * 16 + to_index)
      - 256..319: places ( (cell_index * 4) + piece_idx )  where piece_idx in [0..3] for P,R,N,B

    Most of these will be illegal in a given state; the env simply ignores
    illegal actions and returns a small negative reward so the agent learns
    to avoid them. During training you should still mask illegal actions
    when sampling from the policy.
    """

    def __init__(self):
        self.board: List[Optional[Dict]] = []
        self.pools: Dict[str, List[str]] = {}
        self.current_player: str = PLAYER_X
        self.game_over: bool = False
        self.winner: Optional[str] = None
        self.move_count: int = 0

        # Precompute total action space size
        self.num_actions = 320

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.board = [None] * BOARD_CELLS
        self.pools = {
            PLAYER_X: PIECE_TYPES.copy(),
            PLAYER_O: PIECE_TYPES.copy(),
        }
        self.current_player = PLAYER_X
        self.game_over = False
        self.winner = None
        self.move_count = 0
        return self._encode_state()

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.game_over:
            return self._encode_state(), 0.0, True, {"terminal_already": True}

        legal_actions = self.get_legal_actions()
        action_map = {self._encode_action(a): a for a in legal_actions}

        if action_id not in action_map:
            # Illegal move: small penalty and no state change
            return self._encode_state(), -0.05, False, {"illegal": True}

        action = action_map[action_id]
        self._apply_action(action)
        self.move_count += 1

        winner = self._check_winner(self.board)
        if winner is not None:
            self.game_over = True
            self.winner = winner
            reward = 1.0 if winner == self.current_player else -1.0
            # After win, flip current_player in after_action, but reward is
            # defined from the perspective of the player who just played.
            return self._encode_state(), reward, True, {"winner": winner}

        if self._is_board_full(self.board):
            self.game_over = True
            self.winner = None
            return self._encode_state(), 0.0, True, {"draw": True}

        # Hard cap on game length: if too many moves, declare a draw
        if self.move_count >= 100:
            self.game_over = True
            self.winner = None
            return self._encode_state(), 0.0, True, {"draw": True, "reason": "max_moves"}

        # Non-terminal step: small living penalty to encourage shorter games
        return self._encode_state(), -0.01, False, {}

    # ------------------------------------------------------------------
    # State / observation encoding
    # ------------------------------------------------------------------
    def _encode_state(self) -> np.ndarray:
        """
        Encode board + pools + current_player into a flat vector.

        Board: 16 cells, each one-hot over 9 values:
          [empty, X_P, X_R, X_N, X_B, O_P, O_R, O_N, O_B]
        Pools: counts for each piece type per player -> 2 * 4 = 8 scalars.
        Current player: 1 scalar (1 for X, -1 for O).
        Total size: 16*9 + 8 + 1 = 153.
        """
        board_feats = np.zeros((BOARD_CELLS, 9), dtype=np.float32)
        for idx, cell in enumerate(self.board):
            if cell is None:
                board_feats[idx, 0] = 1.0
            else:
                base = 1 if cell["player"] == PLAYER_X else 5
                offset = {"P": 0, "R": 1, "N": 2, "B": 3}[cell["type"]]
                board_feats[idx, base + offset] = 1.0

        pools_feats = []
        for player in (PLAYER_X, PLAYER_O):
            pool = self.pools[player]
            for t in PIECE_TYPES:
                pools_feats.append(pool.count(t))
        pools_feats = np.array(pools_feats, dtype=np.float32)

        current = np.array([1.0 if self.current_player == PLAYER_X else -1.0], dtype=np.float32)

        return np.concatenate([board_feats.flatten(), pools_feats, current], axis=0)

    # ------------------------------------------------------------------
    # Legal action enumeration and encoding
    # ------------------------------------------------------------------
    def _encode_action(self, action: Action) -> int:
        if action.kind == "move":
            assert action.from_index is not None and action.to_index is not None
            return action.from_index * BOARD_CELLS + action.to_index
        elif action.kind == "place":
            assert action.place_index is not None and action.piece_type is not None
            cell_index = action.place_index
            piece_idx = PIECE_TYPES.index(action.piece_type)
            return 256 + cell_index * len(PIECE_TYPES) + piece_idx
        else:
            raise ValueError(f"Unknown action kind: {action.kind}")

    def get_legal_actions(self) -> List[Action]:
        placements = self._get_all_legal_placements(self.current_player)
        can_move = self._can_player_move_pieces(self.board, self.pools, self.current_player)
        moves = self._get_all_legal_moves(self.current_player) if can_move else []
        return placements + moves

    # ------------------------------------------------------------------
    # Core game logic (ported from game.js)
    # ------------------------------------------------------------------
    def _index_to_row_col(self, index: int) -> Tuple[int, int]:
        row = index // BOARD_SIZE
        col = index % BOARD_SIZE
        return row, col

    def _row_col_to_index(self, row: int, col: int) -> int:
        return row * BOARD_SIZE + col

    def _count_pieces_on_board_for(self, b: List[Optional[Dict]], player: str) -> int:
        return sum(1 for cell in b if cell is not None and cell["player"] == player)

    def _both_players_have_three_for(self, b: List[Optional[Dict]]) -> bool:
        return self._count_pieces_on_board_for(b, PLAYER_X) >= 3 and self._count_pieces_on_board_for(b, PLAYER_O) >= 3

    def _can_player_move_pieces(self, b: List[Optional[Dict]], p: Dict[str, List[str]], player: str) -> bool:
        # Same rule as JS: once both players have at least 3 pieces on board.
        return self._both_players_have_three_for(b)

    # Win / draw checks
    def _check_winner(self, b: List[Optional[Dict]]) -> Optional[str]:
        lines = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
            [0, 5, 10, 15],
            [3, 6, 9, 12],
        ]
        for a, b0, c, d in lines:
            first = b[a]
            if first is None:
                continue
            if (
                b[b0] is not None
                and b[c] is not None
                and b[d] is not None
                and b[b0]["player"] == first["player"]
                and b[c]["player"] == first["player"]
                and b[d]["player"] == first["player"]
            ):
                return first["player"]
        return None

    def _is_board_full(self, b: List[Optional[Dict]]) -> bool:
        return all(cell is not None for cell in b)

    # Move legality helpers
    def _is_legal_rook_move(self, fr: int, fc: int, tr: int, tc: int) -> bool:
        if fr != tr and fc != tc:
            return False
        step_row = 0 if fr == tr else (1 if tr > fr else -1)
        step_col = 0 if fc == tc else (1 if tc > fc else -1)
        r = fr + step_row
        c = fc + step_col
        while r != tr or c != tc:
            idx = self._row_col_to_index(r, c)
            if self.board[idx] is not None:
                return False
            r += step_row
            c += step_col
        return True

    def _is_legal_bishop_move(self, fr: int, fc: int, tr: int, tc: int) -> bool:
        dr = tr - fr
        dc = tc - fc
        if abs(dr) != abs(dc) or dr == 0:
            return False
        step_row = 1 if dr > 0 else -1
        step_col = 1 if dc > 0 else -1
        r = fr + step_row
        c = fc + step_col
        while r != tr or c != tc:
            idx = self._row_col_to_index(r, c)
            if self.board[idx] is not None:
                return False
            r += step_row
            c += step_col
        return True

    def _is_legal_knight_move(self, dr: int, dc: int) -> bool:
        adr = abs(dr)
        adc = abs(dc)
        return (adr == 1 and adc == 2) or (adr == 2 and adc == 1)

    def _is_legal_pawn_move(self, fr: int, fc: int, tr: int, tc: int, piece: Dict) -> bool:
        if "dir" not in piece or not isinstance(piece["dir"], int):
            return False
        dir_ = piece["dir"]
        forward_row = fr - dir_

        # Forward move (no capture)
        if tc == fc and tr == forward_row:
            idx = self._row_col_to_index(tr, tc)
            if self.board[idx] is None:
                return True

        # Diagonal capture
        if tr == forward_row and abs(tc - fc) == 1:
            idx = self._row_col_to_index(tr, tc)
            target = self.board[idx]
            if target is not None and target["player"] != piece["player"]:
                return True

        return False

    def _is_legal_move(self, from_index: int, to_index: int, piece: Dict) -> bool:
        fr, fc = self._index_to_row_col(from_index)
        tr, tc = self._index_to_row_col(to_index)
        dr = tr - fr
        dc = tc - fc
        t = piece["type"]

        if t == "R":
            return self._is_legal_rook_move(fr, fc, tr, tc)
        if t == "B":
            return self._is_legal_bishop_move(fr, fc, tr, tc)
        if t == "N":
            return self._is_legal_knight_move(dr, dc)
        if t == "P":
            return self._is_legal_pawn_move(fr, fc, tr, tc, piece)
        return False

    # Action enumeration
    def _get_all_legal_placements(self, player: str) -> List[Action]:
        results: List[Action] = []
        pool = self.pools[player]
        if not pool:
            return results
        for i in range(BOARD_CELLS):
            if self.board[i] is not None:
                continue
            for t in set(pool):
                # Only consider pieces that are actually still in the pool
                if t in pool:
                    results.append(Action(kind="place", place_index=i, piece_type=t))
        return results

    def _get_all_legal_moves(self, player: str) -> List[Action]:
        moves: List[Action] = []
        for i in range(BOARD_CELLS):
            cell = self.board[i]
            if cell is None or cell["player"] != player:
                continue
            for j in range(BOARD_CELLS):
                if i == j:
                    continue
                target = self.board[j]
                if target is not None and target["player"] == player:
                    continue
                if self._is_legal_move(i, j, cell):
                    moves.append(Action(kind="move", from_index=i, to_index=j))
        return moves

    # Action application
    def _apply_action(self, action: Action) -> None:
        if action.kind == "place":
            assert action.place_index is not None and action.piece_type is not None
            self._apply_placement(self.current_player, action.piece_type, action.place_index)
        elif action.kind == "move":
            assert action.from_index is not None and action.to_index is not None
            self._apply_move(action.from_index, action.to_index)
        else:
            raise ValueError(f"Unknown action kind: {action.kind}")

        # Turn switching and terminal checks are handled in step()
        self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X

    def _apply_placement(self, player: str, piece_type: str, index: int) -> None:
        if self.board[index] is not None:
            return
        pool = self.pools[player]
        if piece_type not in pool:
            return
        pool.remove(piece_type)
        piece = {"player": player, "type": piece_type}
        if piece_type == "P":
            # Note: JS uses dir = -1 for X, +1 for O relative to UI orientation.
            piece["dir"] = -1 if player == PLAYER_X else 1
            if index in [12, 13, 14, 15] and player == PLAYER_O:
                piece["dir"] = -1 * piece["dir"]
            if index in [0, 1, 2, 3] and player == PLAYER_X:
                piece["dir"] = -1 * piece["dir"]
        self.board[index] = piece

    def _apply_move(self, from_index: int, to_index: int) -> None:
        from_cell = self.board[from_index]
        if from_cell is None:
            return
        to_cell = self.board[to_index]
        if to_cell is not None and to_cell["player"] != from_cell["player"]:
            # capture: return piece to opponent's pool
            self.pools[to_cell["player"]].append(to_cell["type"])

        self.board[to_index] = dict(from_cell)
        self.board[from_index] = None

        # Pawn direction reversal at board edge
        if from_cell["type"] == "P" and "dir" in from_cell:
            row, _ = self._index_to_row_col(to_index)
            if row == 0 or row == BOARD_SIZE - 1:
                self.board[to_index]["dir"] = -from_cell["dir"]


def test_env():
    """Quick manual sanity check."""
    env = ChessTicTacToeEnv()
    obs = env.reset()
    print("Initial obs shape:", obs.shape)
    print("Current player:", env.current_player)
    actions = env.get_legal_actions()
    print("Num legal actions at start:", len(actions))
    if actions:
        a_id = env._encode_action(actions[0])
        obs, r, done, info = env.step(a_id)
        print("Step result:", r, done, info)


if __name__ == "__main__":
    test_env()
