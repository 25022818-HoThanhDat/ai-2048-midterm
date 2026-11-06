"""Điểm bắt đầu của bài nộp — sinh viên chỉ cần sửa file này.

Baseline bên dưới nhìn trước đúng một nước. Nó minh họa API và cách mô phỏng
nước đi, nhưng cố ý chưa xử lý ô ngẫu nhiên và chưa phải một thuật toán mạnh.
"""

from __future__ import annotations

import numpy as np

from game2048.observation import Observation
from game2048.rules import move_board, legal_moves
from game2048.types import Move


class Agent:
    # Đổi tên này thành tên đội; nó sẽ xuất hiện trong UI và file kết quả.
    name = "student-agent"

    def __init__(self) -> None:
        """Khởi tạo trạng thái của agent."""
        self.W = {
            "snake": 0.16,
            "empty": 3.,
            "smooth": 0.6,
            "monotonicity": 1.65,
            "merge": 0.7,
            "corner_boost": 20.0,
            "corner_penalty": -30.0,
        }
                
        self.snake_matrix_weight = np.array([
            [15, 14, 13, 12],
            [ 8,  9, 10, 11],
            [ 7,  6,  5,  4],
            [ 0,  1,  2,  3],
        ]).T

        self.reset()

    def reset(self, seed: int | None = None) -> None:
        """Xóa cache/trạng thái giữa hai ván nếu thuật toán của bạn có dùng."""
        self.evaluated_lines = {}
        self.chanced_boards = {}

    def has_valid_merges(self, board: np.ndarray) -> bool:
        """Checks if there are any valid adjacent tile merges available on a full board."""
        # Check horizontal adjacent matches
        if np.any(board[:, :-1] == board[:, 1:]):
            return True
        # Check vertical adjacent matches
        if np.any(board[:-1, :] == board[1:, :]):
            return True
        return False

    def to_log2_board(self, board: np.ndarray) -> np.ndarray:
        """Converts a 2048 game board to log2 tile exponents.
        Empty/invalid cells (<= 0 or NaN) remain 0.0.
        """
        log2_board = np.zeros_like(board, dtype=float)
        valid_mask = (board > 0) & (~np.isnan(board))
        log2_board[valid_mask] = np.int8(np.log2(board[valid_mask]))
        return log2_board

    def evaluate_line(self, line: np.ndarray) -> tuple[float, float]:
        """Calculates (monotonicity, smoothness) penalties for a single row or column."""
        # Extract non-zero tile values
        active_tiles = line[line > 0]

        # Cast to Python ints to prevent float precision duplicates in dict keys
        cache_key = tuple(round(x) for x in active_tiles)

        # 1. Return cached result if available
        if cache_key in self.evaluated_lines:
            return self.evaluated_lines[cache_key]

        # 2. Compute penalties if line has fewer than 2 active tiles
        if len(active_tiles) <= 1:
            result = (0.0, 0.0)
            self.evaluated_lines[cache_key] = result
            return result

        # 3. Calculate Monotonicity & Smoothness
        diffs = np.diff(active_tiles)

        left_to_right = np.sum(diffs[diffs > 0])
        right_to_left = np.sum(-diffs[diffs < 0])
        monotonicity_penalty = -float(min(left_to_right, right_to_left))

        smoothness_penalty = -float(np.sum(np.abs(diffs)))

        result = (monotonicity_penalty, smoothness_penalty)

        # 4. Save to cache and return
        self.evaluated_lines[cache_key] = result
        return result
    
    def calculate_board_metrics(self, log2_board: np.ndarray, chance_cal: bool = False) -> tuple[float, float]:
        """Calculates total (monotonicity, smoothness) penalties for the log2 board."""
        total_monotonicity = 0.0
        total_smoothness = 0.0

        # Total empty cells across the entire board (y)
        total_empty_board = int(np.count_nonzero(log2_board == 0))

        # Collect 4 rows and 4 columns
        lines = [log2_board[j, :] for j in range(4)] + [log2_board[:, k] for k in range(4)]

        for line in lines:
            empty_indices = np.where(line == 0)[0]
            x = len(empty_indices)

            if not chance_cal or x == 0 or total_empty_board == 0:
                mono, smooth = self.evaluate_line(line)
                total_monotonicity += mono
                total_smoothness += smooth
            else:
                # 1. Base evaluation when no block appears in this line
                mono_none, smooth_none = self.evaluate_line(line)
                p_no_spawn = 1.0 - (x / total_empty_board)
                
                line_mono = p_no_spawn * mono_none
                line_smooth = p_no_spawn * smooth_none

                # P(Spawn at a specific spot i) = (x / total_empty_board) * (1 / x) = 1 / total_empty_board
                p_spot = 1.0 / total_empty_board

                for idx in empty_indices:
                    # 2. Block '2' appears (log2(2) = 1)
                    line_with_2 = line.copy()
                    line_with_2[idx] = np.int8(1)
                    mono_2, smooth_2 = self.evaluate_line(line_with_2)

                    # 3. Block '4' appears (log2(4) = 2)
                    line_with_4 = line.copy()
                    line_with_4[idx] = np.int8(2)
                    mono_4, smooth_4 = self.evaluate_line(line_with_4)

                    # 90% chance of 2, 10% chance of 4
                    expected_mono = 0.9 * mono_2 + 0.1 * mono_4
                    expected_smooth = 0.9 * smooth_2 + 0.1 * smooth_4

                    # Accumulate per-spot expectation into line totals
                    line_mono += p_spot * expected_mono
                    line_smooth += p_spot * expected_smooth

                # Add the fully computed line total to board totals ONCE per line
                total_monotonicity += line_mono
                total_smoothness += line_smooth

        return total_monotonicity, total_smoothness

    def calculate_snake_val(self, log2_board: np.ndarray, chance_cal: bool = False) -> float:
        """Calculates the snake matrix heuristic value for the log2 board."""
        total_empty_board = int(np.count_nonzero(log2_board == 0))

        if not chance_cal or total_empty_board == 0:
            return float(np.sum(log2_board * self.snake_matrix_weight))

        #empty_coords = np.argwhere(log2_board == 0)
        # Fast (Pure Python tuple iteration on flat board):
        flat_board = log2_board.ravel()
        empty_indices = [i for i, val in enumerate(flat_board) if val == 0]
        #num_empty = len(empty_indices)
        
        base_score = float(np.sum(log2_board * self.snake_matrix_weight))
        p_spot = 1.0 / total_empty_board
        expected_snake_val = 0.0

        for idx in empty_indices:
            r, c = np.unravel_index(idx, log2_board.shape)
            weight = self.snake_matrix_weight[r, c]
            expected_tile_exponent = 1.1
            spot_spawn_score = base_score + (expected_tile_exponent * weight)
            expected_snake_val += p_spot * spot_spawn_score

        return float(expected_snake_val)
    
    def evaluate(self, board: np.ndarray, merge_score: int, time_remaining_ms: float = float('inf')) -> dict[str, float]:
        """Evaluates a board state and returns a dictionary of weighted scores."""
        log2_board = self.to_log2_board(board)

        # Convert board into a flat hashable tuple of integers for caching
        board_key = board.tobytes()
        #board_key = tuple(log2_board.astype(int).flat)

        # 1. Retrieve structural component scores from cache if present
        if board_key in self.chanced_boards:
            cached_components = self.chanced_boards[board_key]
        else:
            # Calculate heuristics (chance_cal flag condition can be toggled via time_remaining_ms if needed)
            use_chance = (time_remaining_ms > 500.0)
            
            snake_val = self.calculate_snake_val(log2_board, chance_cal=use_chance)
            
            # Empty cells heuristic based on raw board
            empty_cells = int(np.count_nonzero(board == 0) - 1)
            
            monotonicity_val, smoothness_val = self.calculate_board_metrics(log2_board, chance_cal=use_chance)

            # Corner bonus check
            max_pos = np.unravel_index(np.argmax(board), board.shape)
            if max_pos == (0, 0):  # Top-left corner
                corner_bonus = self.W["corner_boost"]  
            else:
                corner_bonus = self.W["corner_penalty"] 

            # Store weighted layout metrics in cache
            cached_components = {
                "snake": snake_val * self.W["snake"],
                "empty": empty_cells * self.W["empty"],
                "monotonicity": monotonicity_val * self.W["monotonicity"],
                "smooth": smoothness_val * self.W["smooth"],
                "corner": corner_bonus,
            }
            self.chanced_boards[board_key] = cached_components

        # 2. Add dynamic merge_score metric for the current move
        merge_val = np.log2(max(1, merge_score))
        
        scores = dict(cached_components)
        scores["merge"] = float(merge_val * self.W["merge"])
        scores["total"] = sum(scores.values())

        return scores

    def chanced_evaluate(
        self, 
        board: np.ndarray, 
        merge_score: int, 
        time_remaining_ms: float, 
        depth: int,
    ) -> dict[str, float]:
        """Chance Node: Computes expected evaluation across prospective tile spawns."""
        
        if depth <= 0:
            return self.evaluate(board, merge_score, time_remaining_ms)

        #empty_coords = np.argwhere(board == 0)

        # Fast (Pure Python tuple iteration on flat board):
        flat_board = board.ravel()
        empty_indices = [i for i, val in enumerate(flat_board) if val == 0]
        num_empty = len(empty_indices)
        #num_empty = len(empty_coords)

        # Terminal check
        if num_empty == 0 and not self.has_valid_merges(board):
            return {
                "snake": 0.0,
                "empty": -100000.0,
                "merge": 0.0,
                "monotonicity": 0.0,
                "smooth": 0.0,
                "corner": 0.0,
                "total": -100000.0
            }

        if num_empty == 0:
            return self.evaluate(board, merge_score, time_remaining_ms)

        

        base_eval = self.evaluate(board, merge_score, time_remaining_ms)
        p_spot = 1.0 / num_empty
        expected_total = 0.0
        eval_fours = (depth <= 1)

        for i, idx in enumerate(empty_indices, 1):
            r, c = np.unravel_index(idx, board.shape)
            # 1. Simulate spawn of '2'
            board_with_2 = board.copy()
            board_with_2[r, c] = 2
            res_2, _ = self.move_step(board_with_2, time_remaining_ms, depth - 1)

            if eval_fours:
                # 2. Simulate spawn of '4'
                board_with_4 = board.copy()
                board_with_4[r, c] = 4
                res_4, _ = self.move_step(board_with_4, time_remaining_ms, depth - 1)
                
                spot_expected = 0.9 * res_2["total"] + 0.1 * res_4["total"]
            else:
                spot_expected = res_2["total"]

            expected_total += p_spot * spot_expected

        result = dict(base_eval)
        result["total"] = expected_total
        return result

    def move_step(
        self, 
        current_board: np.ndarray, 
        time_remaining_ms: float, 
        depth: int,
    ) -> tuple[dict[str, float], Move | None]:
        """Max Node: Evaluates candidate directions and returns best (evaluation_dict, Move)."""
        best_eval: dict[str, float] | None = None
        best_move: Move | None = None
        max_total = -float("inf")

        for move in Move:
            next_board, merge_score, changed = move_board(current_board, move)
            if not changed:
                continue

            expected_eval = self.chanced_evaluate(
                next_board, 
                merge_score, 
                time_remaining_ms, 
                depth - 1
            )

            if expected_eval["total"] > max_total:
                max_total = expected_eval["total"]
                best_eval = expected_eval
                best_move = move

        if best_move is None or best_eval is None:
            terminal_eval = {
                "snake": 0.0,
                "empty": -100000.0,
                "merge": 0.0,
                "monotonicity": 0.0,
                "smooth": 0.0,
                "corner": 0.0,
                "total": -100000.0
            }
            return terminal_eval, None

        return best_eval, best_move
    
    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        """Entry point called by the game runner to pick a move."""
        if not obs.legal_moves:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")

        # Early game script to quickly build into a corner
        if obs.move_count <= 5:
            stmo = Move.UP if Move.UP in obs.legal_moves else Move.LEFT if Move.LEFT in obs.legal_moves else obs.legal_moves[0]
            return stmo

        # 1. Determine base depth using the amount of empty cells
        num_empty = int(np.count_nonzero(obs.board == 0))

        tight_time_penalty = 0
        if obs.total_time_limit_ms <= 10000.0:
            # Short total time limit: reduce depth to avoid timeouts
            tight_time_penalty = 1

        elif num_empty <= 1:
            search_depth = 4
        elif num_empty <= 2:
            search_depth = 3
        elif num_empty <= 4:
            search_depth = 2
        elif num_empty <= 12:
            search_depth = 1
        else:
            # Early game / sparse board: large branching factor, stick to depth 1
            search_depth = 0

        # 2. Time management check
        # Force a shallower search if total game time is running out
        if obs.time_remaining_ms < 500.0:
            # Panic mode: Less than 1 second left for the whole game
            search_depth = 1
        elif obs.time_remaining_ms < 5000.0:
            # Low time: Less than 5 seconds left
            search_depth = min(search_depth, 2)

        search_depth = max(0, search_depth - tight_time_penalty)

        print("----------------------------------")
        print("current board\n", obs.board)
        print(f"legal moves: {[move.name for move in obs.legal_moves]}")
        print(f"Empty cells: {num_empty} | Time remaining: {obs.time_remaining_ms:.0f}ms | Chosen Depth: {search_depth}")

        # Start search tree with dynamic depth
        best_eval_dict, action_to_make = self.move_step(
            obs.board, 
            obs.time_remaining_ms, 
            depth=search_depth
        )

        # Fallback if tree search surprisingly returns None or an invalid move
        if action_to_make is None or action_to_make not in obs.legal_moves:
            action_to_make = obs.legal_moves[0]

        print(f"Best move {action_to_make.name} and best evaluation breakdown: {best_eval_dict}")

        return action_to_make