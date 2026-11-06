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

    def reset(self, seed: int | None = None) -> None:
        """Xóa cache/trạng thái giữa hai ván nếu thuật toán của bạn có dùng."""
        self.evaluated_lines.clear()
        self.evaluated_boards.clear()

    def __init__(self) -> None:
        """Khởi tạo trạng thái của agent. Không cần thiết phải reset() ở đây."""
        self.evaluated_lines = {
        
        }
        # list of tuple of (monotonicity_penalty, smoothness_penalty) for each evaluated lines
        self.evaluated_boards = {

        }
        self.chanced_boards = {

        }
        self.reset()
        self.W = {
                    "snake": 0.15,
                    "empty": 3.0,
                    "smooth": 0.5,
                    "monotonicity": 1.5,
                    "merge": 0.5,
                    "corner_boost": 20.0,
                    "corner_penalty": -20.0,
                }
                
        self.snake_matrix_weight = np.array([
                    [15, 14, 13, 12],
                    [ 8,  9, 10, 11],
                    [ 7,  6,  5,  4],
                    [ 0,  1,  2,  3],
                ]).T

    @staticmethod
    def to_log2_board(self, board: np.ndarray) -> np.ndarray:
        """Converts a 2048 game board to log2 tile exponents.
        
        Empty/invalid cells (<= 0 or NaN) remain 0.0.
        """
        log2_board = np.zeros_like(board, dtype=float)
        valid_mask = (board > 0) & (~np.isnan(board))
        #print(valid_mask, board, log2_board)
        log2_board[valid_mask] = np.int8(np.log2(board[valid_mask]))
        #print(log2_board)
        return log2_board

    def evaluate_line(self, line: np.ndarray) -> tuple[float, float]:
        """ASSUME LINE IS LOG2 LINES
        Calculates (monotonicity, smoothness) penalties for a single row or column.
        
        Ignores empty cells (0s) and caches results in self.evaluated_lines.


        if not hasattr(self, "evaluated_lines"):
            self.evaluated_lines: dict[tuple[int, ...], tuple[float, float]] = {}
        """
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
        """Calculates total (monotonicity, smoothness) penalties for the log2 board

        by delegating line evaluations to the cached evaluate_line method.
        """
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
        """Calculates the snake matrix heuristic value for the log2 board.

        If chance_cal is True, computes the expected snake score by factoring in
        probabilistic tile spawns (2 with 90% chance, 4 with 10% chance) at open cells.
        """
        total_empty_board = int(np.count_nonzero(log2_board == 0))

        if not chance_cal or total_empty_board == 0:
            return float(np.sum(log2_board * self.snake_matrix_weight))

        empty_coords = np.argwhere(log2_board == 0)
        
        # 1. Base score for the current board state (No spawn case)
        # P(No Spawn anywhere) is 0, but for cell-wise probability partitioning:
        # Each specific empty cell has probability P(Spawn at cell) = 1 / total_empty_board
        base_score = float(np.sum(log2_board * self.snake_matrix_weight))
        
        # P(No spawn at a given empty spot) = 1 - (1 / y)
        p_spot = 1.0 / total_empty_board

        expected_snake_val = 0.0

        for r, c in empty_coords:
            weight = self.snake_matrix_weight[r, c]

            # Expected value added if a tile spawns at (r, c):
            # 90% chance of log2(2) = 1.0, 10% chance of log2(4) = 2.0
            # Expected contribution = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
            expected_tile_exponent = 1.1
            spot_spawn_score = base_score + (expected_tile_exponent * weight)

            expected_snake_val += p_spot * spot_spawn_score

        return float(expected_snake_val)
    
    @staticmethod
    def evaluate(self, board: np.ndarray, merge_score: int, time_remaining_ms: float = float('inf')) -> dict[str, float]:
        """Evaluates a board state and returns a dictionary of weighted scores.
        
        Caches board layout metrics in self.chanced_boards to avoid recalculating 
        structural heuristics for identical board states.

        if not hasattr(self, "chanced_boards"):
                    self.chanced_boards: dict[tuple[int, ...], dict[str, float]] = {}
        """

        log2_board = self.to_log2_board(self,board)

        # Convert board into a flat hashable tuple of integers for caching
        board_key = tuple(log2_board.astype(int).flat)

        # 1. Retrieve structural component scores from cache if present
        if board_key in self.chanced_boards:
            cached_components = self.chanced_boards[board_key]
        else:
            # Calculate heuristics (chance_cal flag condition can be toggled via time_remaining_ms if needed)
            use_chance = (time_remaining_ms > 2500.0)
            
            snake_val = self.calculate_snake_val(log2_board, chance_cal=use_chance)
            
            # Empty cells heuristic based on raw board
            empty_cells = int(np.count_nonzero(board == 0) - 1)
            
            monotonicity_val, smoothness_val = self.calculate_board_metrics(log2_board, chance_cal=use_chance)

            # Corner bonus check
            max_pos = np.unravel_index(np.argmax(board), board.shape)
            if max_pos == (0, 0):  # Top-left corner
                corner_bonus = self.W["corner_boost"]  # Reward for keeping max tile in corner
            else:
                corner_bonus = self.W["corner_penalty"]  # Penalty for pulling max tile out of corner

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
        _indent: int = 0
    ) -> dict[str, float]:
        """Chance Node: Computes expected evaluation across prospective tile spawns."""
        prefix = "  " * _indent
        #print(f"{prefix}[CHANCE] Entering Depth {depth} | Empty cells: {np.count_nonzero(board == 0)}")

        # Fix: Correct self method call (removed extra 'self' argument)
        if depth <= 0:
            val = self.evaluate(self, board, merge_score, time_remaining_ms)
            #print(f"{prefix}[LEAF] Base depth reached (0). Returning total score: {val['total']:.2f}")
            return val

        empty_coords = np.argwhere(board == 0)
        num_empty = len(empty_coords)

        # Terminal check
        if num_empty == 0 and not self.has_valid_merges(board):
            #print(f"{prefix}[TERMINAL] Full board with no valid merges detected!")
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
            val = self.evaluate(self, board, merge_score, time_remaining_ms)
            #print(f"{prefix}[CHANCE] Full board with available merges. Score: {val['total']:.2f}")
            return val

        base_eval = self.evaluate(self, board, merge_score, time_remaining_ms)
        p_spot = 1.0 / num_empty
        expected_total = 0.0
        eval_fours = (depth <= 1)

        #print(f"{prefix}[CHANCE] Simulating {num_empty} spawn slots (eval_fours={eval_fours})...")
        
        #start_t = time.perf_counter()

        for i, (r, c) in enumerate(empty_coords, 1):
            # 1. Simulate spawn of '2'
            board_with_2 = board.copy()
            board_with_2[r, c] = 2
            res_2, _ = self.move_step(board_with_2, time_remaining_ms, depth - 1, _indent=_indent + 1)

            if eval_fours:
                # 2. Simulate spawn of '4'
                board_with_4 = board.copy()
                board_with_4[r, c] = 4
                res_4, _ = self.move_step(board_with_4, time_remaining_ms, depth - 1, _indent=_indent + 1)
                
                spot_expected = 0.9 * res_2["total"] + 0.1 * res_4["total"]
            else:
                spot_expected = res_2["total"]

            expected_total += p_spot * spot_expected

        #elapsed_ms = (time.perf_counter() - start_t) * 1000
        #print(f"{prefix}[CHANCE] Depth {depth} completed in ms | Expected Total: {expected_total:.2f}")

        result = dict(base_eval)
        result["total"] = expected_total
        return result


    def move_step(
        self, 
        current_board: np.ndarray, 
        time_remaining_ms: float, 
        depth: int,
        _indent: int = 0
    ) -> tuple[dict[str, float], Move | None]:
        """Max Node: Evaluates candidate directions and returns best (evaluation_dict, Move)."""
        #prefix = "  " * _indent
        #print(f"{prefix}[MAX] Evaluating candidate moves at Depth {depth}...")

        best_eval: dict[str, float] | None = None
        best_move: Move | None = None
        max_total = -float("inf")
        evaluated_moves = 0

        for move in Move:
            next_board, merge_score, changed = move_board(current_board, move)
            if not changed:
                #print(f"{prefix}  -> Move {move.name}: Invalid/No change (Skipped)")
                continue

            evaluated_moves += 1
            #print(f"{prefix}  -> Move {move.name}: Valid! Branching into Chance node...")

            # FIX: Decremented depth to depth - 1 to stop infinite recursion!
            expected_eval = self.chanced_evaluate(
                next_board, 
                merge_score, 
                time_remaining_ms, 
                depth - 1, 
                #_indent=_indent + 1
            )

            #print(f"{prefix}  -> Move {move.name} evaluated total score: {expected_eval['total']:.2f}")

            if expected_eval["total"] > max_total:
                max_total = expected_eval["total"]
                best_eval = expected_eval
                best_move = move

        if best_move is None or best_eval is None:
            #print(f"{prefix}[MAX] Game Over! No valid moves found at Depth {depth}.")
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

        #print(f"{prefix}[MAX] Depth {depth} Best Choice: Move {best_move.name} with score {max_total:.2f}")
        return best_eval, best_move
    
    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        """Trả về đúng một phần tử ``Move`` có trong ``obs.legal_moves``.

        Khi chấm theo tổng thời gian, ``time_limit_ms`` và
        ``obs.time_remaining_ms`` là ngân sách còn lại của cả ván — không phải
        lời hứa rằng mỗi nước được phép dùng hết số đó.
        """
        if not obs.legal_moves:
            raise ValueError("choose_move được gọi ở trạng thái kết thúc")

        if obs.move_count <= 5:
            stmo = Move.UP if Move.UP in obs.legal_moves else Move.LEFT if Move.LEFT in obs.legal_moves else obs.legal_moves[0]
            return stmo

        #best_move = obs.legal_moves[0]
        #best_value = float("-inf")

        print("----------------------------------")
        print("current board\n",obs.board)
        print(f"legal moves: {[move.name for move in obs.legal_moves]}")

        best_eval_dict, action_to_make = self.move_step(
            obs.board, 
            obs.time_remaining_ms, 
            depth=2
        )

        print(f"Best move {action_to_make.name} and best evaluation breakdown: {best_eval_dict}")

        # Execute action_to_make (e.g., Move.UP, Move.LEFT, etc.)
        return action_to_make
