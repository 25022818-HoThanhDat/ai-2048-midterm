"""Điểm bắt đầu của bài nộp — sinh viên chỉ cần sửa file này.

Baseline bên dưới nhìn trước đúng một nước. Nó minh họa API và cách mô phỏng
nước đi, nhưng cố ý chưa xử lý ô ngẫu nhiên và chưa phải một thuật toán mạnh.
"""

from __future__ import annotations

import numpy as np

from game2048.observation import Observation
from game2048.rules import move_board
from game2048.types import Move


class Agent:
    # Đổi tên này thành tên đội; nó sẽ xuất hiện trong UI và file kết quả.
    name = "student-agent"

    def reset(self, seed: int | None = None) -> None:
        """Xóa cache/trạng thái giữa hai ván nếu thuật toán của bạn có dùng."""

    #@staticmethod
    def calculate_monotonicity(log2_board: np.ndarray) -> float:
        """Calculates monotonicity while ignoring empty spaces (0s)."""
        total_monotonicity = 0.0

        # 1. Check Row Monotonicity
        for r in range(4):
            # Filter out 0s to get only active tiles in this row
            row = log2_board[r, log2_board[r, :] > 0]
            if len(row) > 1:
                diffs = np.diff(row)
                left_to_right = np.sum(diffs[diffs > 0])
                right_to_left = np.sum(-diffs[diffs < 0])
                total_monotonicity -= min(left_to_right, right_to_left)

        # 2. Check Column Monotonicity
        for c in range(4):
            # Filter out 0s to get only active tiles in this column
            col = log2_board[log2_board[:, c] > 0, c]
            if len(col) > 1:
                diffs = np.diff(col)
                top_to_bottom = np.sum(diffs[diffs > 0])
                bottom_to_top = np.sum(-diffs[diffs < 0])
                total_monotonicity -= min(top_to_bottom, bottom_to_top)

        return total_monotonicity

    def calculate_smoothness(log2_board: np.ndarray) -> float:
        """Calculates smoothness while ignoring empty spaces (0s)."""
        smoothness = 0.0

        # Check Row Smoothness
        for r in range(4):
            row = log2_board[r, log2_board[r, :] > 0]
            if len(row) > 1:
                diffs = np.diff(row)
                smoothness -= np.sum(np.abs(diffs))

        # Check Column Smoothness
        for c in range(4):
            col = log2_board[log2_board[:, c] > 0, c]
            if len(col) > 1:
                diffs = np.diff(col)
                smoothness -= np.sum(np.abs(diffs))

        return smoothness

    @staticmethod
    def evaluate(board: np.ndarray, merge_score: int, time_remaining_ms: float = float('inf')) -> dict[str, float]:
        W = {
            "snake": 0.1,
            "empty": 3.0,
            "smooth": 0.5,
            "monotonicity": 1.0,
            "merge": 0.5,
            "corner": 1.0,
        }
        
        snake_matrix_weight = np.array([
            [15, 14, 13, 12],
            [ 8,  9, 10, 11],
            [ 7,  6,  5,  4],
            [ 0,  1,  2,  3],
        ])
        
        log2_board = np.zeros_like(board, dtype=float)
        valid_mask = (board > 0) & (~np.isnan(board))
        log2_board[valid_mask] = np.log2(board[valid_mask])

        snake_val = np.linalg.norm(log2_board * snake_matrix_weight, ord=1)
        empty_cells = int(np.count_nonzero(board == 0))
        #largest_tile = int(board.max()) if board.size > 0 else 0
        monotonicity_val = Agent.calculate_monotonicity(log2_board)
        smoothness_val = Agent.calculate_smoothness(log2_board)
        merge_val = np.log2(max(1, merge_score))

        #corner bonus: check if the largest tile is in a corner
        max_pos = np.unravel_index(np.argmax(board), board.shape)
        if max_pos == (0, 0):  # Top-left corner
            corner_bonus = 10.0
        else:
            corner_bonus = -5.0  # Heavy penalty for pulling max tile out of corner


        scores = {
            "snake": snake_val * W["snake"],
            "empty": empty_cells * W["empty"],
            #"largest": np.log2(largest_tile) * W["largest"], # Tip: Consider using np.log2(largest_tile) here!
            "merge": merge_val * W["merge"],
            "monotonicity": monotonicity_val * W["monotonicity"],
            "smooth": smoothness_val * W["smooth"],
            "corner": corner_bonus * W["corner"],
        }
        scores["total"] = sum(scores.values())
        return scores
        #return 250.0 * empty_cells + merge_score + 0.1 * largest_tile

    def expectimax_step(self, next_board: np.ndarray, merge_score: int, time_remaining_ms: float) -> dict[str, float]:
        """Evaluates board state using an Expectimax chance node if board space is tight

        and remaining time permits. Returns a detailed dictionary breakdown.
        """
        empty_coords = np.argwhere(next_board == 0)
        num_empty = len(empty_coords)

        # Condition: Half or less board empty (<= 12 empty cells) AND time > 2500ms
        if num_empty <= 12 and time_remaining_ms > 2500.0 and num_empty > 0:
            cell_prob = 1.0 / num_empty

            # 1. Evaluate first spawn to get dictionary keys
            board_2_init = next_board.copy()
            board_2_init[empty_coords[0][0], empty_coords[0][1]] = 2
            sample_dict = self.evaluate(board_2_init, merge_score)

            # 2. Initialize expected_value dict with 0.0 for all keys
            expected_value = {key: 0.0 for key in sample_dict.keys()}

            # 3. Accumulate weighted scores across all empty cells
            for r, c in empty_coords:
                board_with_2 = next_board.copy()
                board_with_2[r, c] = 2
                val_2 = self.evaluate(board_with_2, merge_score)

                board_with_4 = next_board.copy()
                board_with_4[r, c] = 4
                val_4 = self.evaluate(board_with_4, merge_score)

                # Element-wise addition for every component key
                for key in expected_value:
                    expected_value[key] += cell_prob * (0.9 * val_2[key] + 0.1 * val_4[key])

            # 4. Set depth metadata tag
            expected_value["expectimax_max_depth"] = 1
            return expected_value

        # Direct evaluation (depth = 0)
        res_dict = self.evaluate(next_board, merge_score)
        res_dict["expectimax_max_depth"] = 0
        return res_dict
    
    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        """Trả về đúng một phần tử ``Move`` có trong ``obs.legal_moves``.

        Khi chấm theo tổng thời gian, ``time_limit_ms`` và
        ``obs.time_remaining_ms`` là ngân sách còn lại của cả ván — không phải
        lời hứa rằng mỗi nước được phép dùng hết số đó.
        """
        if not obs.legal_moves:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")

        best_move = obs.legal_moves[0]
        best_value = float("-inf")

        print("----------------------------------")
        print(f"legal moves: {[move.name for move in obs.legal_moves]}")

        for move in obs.legal_moves:
            next_board, merge_score, _changed = move_board(obs.board, move)
            print(f"Next board for move {move.name}:\n{next_board}\nMerge score: {merge_score}")

            value = self.expectimax_step(next_board, merge_score, obs.time_remaining_ms)
            if isinstance(value, float):
                value = {"total": value}

            print(f"Move {move.name}: Total={value['total']:.2f} | Breakdown: {value}")
            
            if value["total"] > best_value:
                best_value, best_move = value["total"], move
        return best_move


