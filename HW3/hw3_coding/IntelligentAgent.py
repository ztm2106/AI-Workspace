# 10 game Test
# 4096
# 1024
# 512
# 1024
# 1024
# 2048
# 2048
# 1024
# 2048
# 512

# {4096, 2048, 2048, 2048, 1024}

# reached 2048 consistently and 4096
import random
import time
import math
from BaseAI import BaseAI

class IntelligentAgent(BaseAI):
    def __init__(self):
        self.time_limit = 0.2  # Maximum time allowed per move
        self.start_time = None
        self.depth_limit = 4  # Starting depth limit, will increase adaptively

    def getMove(self, grid):
        """Selects the best move by running expectiminimax with alpha-beta pruning."""
        self.start_time = time.process_time()
        best_move, best_score = None, float('-inf')

        # Adaptive depth control: start at 2, increase as long as time permits
        self.depth_limit = 4
        while not self.is_time_exceeded():
            for move in grid.getAvailableMoves():
                new_grid = grid.clone()
                new_grid.move(move[0])
                move_score = self.expectiminimax(new_grid, depth=0, is_maximizing=False, alpha=float('-inf'), beta=float('inf'))

                if move_score > best_score:
                    best_score, best_move = move_score, move[0]

            # Increase depth limit more aggressively if ample time remains
            if not self.is_time_exceeded() and (time.process_time() - self.start_time) < self.time_limit / 2:
                self.depth_limit += 1
            else:
                break

        # Return the best move, or a random move if none are available
        return best_move if best_move is not None else random.choice(grid.getAvailableMoves())[0]

    def expectiminimax(self, grid, depth, is_maximizing, alpha, beta):
        """Runs expectiminimax with alpha-beta pruning to evaluate moves."""
        if self.is_time_exceeded() or not grid.getAvailableMoves() or depth >= self.depth_limit:
            return self.evaluate_grid(grid)
        
        if is_maximizing:
            max_score = float('-inf')
            for move in grid.getAvailableMoves():
                new_grid = grid.clone()
                new_grid.move(move[0])
                score = self.expectiminimax(new_grid, depth + 1, False, alpha, beta)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_score
        else:
            min_score = 0
            empty_cells = grid.getAvailableCells()
            prob_2, prob_4 = 0.9, 0.1
            
            for cell in empty_cells:
                for tile_value, probability in [(2, prob_2), (4, prob_4)]:
                    new_grid = grid.clone()
                    new_grid.insertTile(cell, tile_value)
                    min_score += self.expectiminimax(new_grid, depth + 1, True, alpha, beta) * probability

                beta = min(beta, min_score)
                if beta <= alpha:
                    break

            return min_score / len(empty_cells) if empty_cells else 0

    def evaluate_grid(self, grid):
        """Evaluates the grid using monotonicity, empty cells, and a corner bias for max tile."""
        monotonicity = self.calculate_monotonicity(grid)
        empty_cells = len(grid.getAvailableCells())
        corner_bias = self.corner_max_tile(grid)
        
        # Adjusted heuristic weights for improved performance
        return monotonicity * 3 + empty_cells * 2 + corner_bias * 1.5

    def is_time_exceeded(self):
        """Checks if the time limit for the move has been exceeded."""
        return (time.process_time() - self.start_time) >= self.time_limit

    def calculate_monotonicity(self, grid):
        """Calculates a monotonicity score that penalizes rows/columns that are not ordered."""
        monotonicity = 0

        # Row and column monotonicity, prioritizing left and top side for higher values
        for row in grid.map:
            for i in range(3):
                # Penalize deviations from a left-to-right descending order in each row
                if row[i] < row[i + 1]:
                    monotonicity -= abs(row[i] - row[i + 1])

        for col in range(4):
            for i in range(3):
                # Penalize deviations from a top-to-bottom descending order in each column
                if grid.map[i][col] < grid.map[i + 1][col]:
                    monotonicity -= abs(grid.map[i][col] - grid.map[i + 1][col])

        return monotonicity

    def corner_max_tile(self, grid):
        """Adds a bonus if the max tile is in the top-left corner, encouraging stable positioning."""
        max_tile = self.get_max_tile(grid)
        if grid.map[0][0] == max_tile:
            return 2.5 * max_tile
        return 0

    def get_max_tile(self, grid):
        """Returns the highest tile value on the grid."""
        return max(max(row) for row in grid.map)


