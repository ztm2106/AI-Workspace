# import random
# import time
# from BaseAI import BaseAI

# class IntelligentAgent(BaseAI):
#     def __init__(self):
#         self.time_limit = 0.2  # Maximum time allowed per move
#         self.start_time = None
    
#     def getMove(self, grid):
#         """Selects the best move by running expectiminimax with alpha-beta pruning."""
#         self.start_time = time.process_time()
#         best_move, best_score = None, float('-inf')
        
#         for move in grid.getAvailableMoves():
#             new_grid = grid.clone()
#             new_grid.move(move[0])
#             move_score = self.expectiminimax(new_grid, depth=0, is_maximizing=False, alpha=float('-inf'), beta=float('inf'))
            
#             if move_score > best_score:
#                 best_score, best_move = move_score, move[0]
        
#         # Return the best move, or a random move if none are available
#         return best_move if best_move is not None else random.choice(grid.getAvailableMoves())[0]
    
#     def expectiminimax(self, grid, depth, is_maximizing, alpha, beta):
#         """Runs expectiminimax with alpha-beta pruning to evaluate moves."""
#         if self.is_time_exceeded() or not grid.getAvailableMoves() or depth >= 3:
#             return self.evaluate_grid(grid)
        
#         if is_maximizing:
#             max_score = float('-inf')
#             for move in grid.getAvailableMoves():
#                 new_grid = grid.clone()
#                 new_grid.move(move[0])
#                 score = self.expectiminimax(new_grid, depth + 1, False, alpha, beta)
#                 max_score = max(max_score, score)
#                 alpha = max(alpha, score)
#                 if beta <= alpha:
#                     break
#             return max_score
#         else:
#             min_score = 0
#             empty_cells = grid.getAvailableCells()
#             prob_2, prob_4 = 0.9, 0.1
            
#             for cell in empty_cells:
#                 # Evaluate placement of a 2 tile
#                 grid_2 = grid.clone()
#                 grid_2.insertTile(cell, 2)
#                 min_score += self.expectiminimax(grid_2, depth + 1, True, alpha, beta) * prob_2
                
#                 # Evaluate placement of a 4 tile
#                 grid_4 = grid.clone()
#                 grid_4.insertTile(cell, 4)
#                 min_score += self.expectiminimax(grid_4, depth + 1, True, alpha, beta) * prob_4
                
#                 beta = min(beta, min_score)
#                 if beta <= alpha:
#                     break
            
#             return min_score / len(empty_cells) if empty_cells else 0

#     def evaluate_grid(self, grid):
#         """Evaluates the grid using weighted heuristics."""
#         max_tile = self.get_max_tile(grid)
#         smoothness = self.calculate_smoothness(grid)
#         monotonicity = self.calculate_monotonicity(grid)
#         empty_cells = len(grid.getAvailableCells())
        
#         # Heuristic weights (can be tuned further for performance)
#         return (
#             1.0 * max_tile +
#             1.5 * smoothness +
#             2.0 * monotonicity +
#             3.0 * empty_cells

#             # 5.0 * max_tile +
#             # 2.0 * smoothness +
#             # 0.5 * monotonicity +
#             # 1.0 * empty_cells

#             # got to 2048
#             # 5.0 * max_tile +
#             # 3.0 * smoothness +
#             # 1.5 * monotonicity +
#             # 2.0 * empty_cells
#         )
    
#     def is_time_exceeded(self):
#         """Checks if the time limit for the move has been exceeded."""
#         return (time.process_time() - self.start_time) >= self.time_limit
    
#     def get_max_tile(self, grid):
#         """Returns the highest tile value on the grid."""
#         return max(max(row) for row in grid.map)
    
#     def calculate_smoothness(self, grid):
#         """Calculates a smoothness score, favoring adjacent tiles with similar values."""
#         smoothness = 0
#         for x in range(4):
#             for y in range(4):
#                 if grid.map[x][y] != 0:
#                     value = grid.map[x][y]
#                     # Check right and down neighbors
#                     if x + 1 < 4 and grid.map[x + 1][y] != 0:
#                         smoothness -= abs(value - grid.map[x + 1][y])
#                     if y + 1 < 4 and grid.map[x][y + 1] != 0:
#                         smoothness -= abs(value - grid.map[x][y + 1])
#         return smoothness
    
#     def calculate_monotonicity(self, grid):
#         """Calculates a monotonicity score, favoring rows and columns with values in decreasing order."""
#         score = 0
#         # Row monotonicity
#         for row in grid.map:
#             for i in range(3):
#                 if row[i] < row[i + 1]:
#                     score += row[i + 1] - row[i]
        
#         # Column monotonicity
#         for col in range(4):
#             for i in range(3):
#                 if grid.map[i][col] < grid.map[i + 1][col]:
#                     score += grid.map[i + 1][col] - grid.map[i][col]
#         return score











# import random
# import time
# import math
# from BaseAI import BaseAI

# class IntelligentAgent(BaseAI):
#     def __init__(self):
#         self.time_limit = 0.2  # Maximum time allowed per move
#         self.start_time = None
#         self.depth_limit = 2  # Starting depth limit, will increase adaptively

#     def getMove(self, grid):
#         """Selects the best move by running expectiminimax with alpha-beta pruning."""
#         self.start_time = time.process_time()
#         best_move, best_score = None, float('-inf')

#         # Adaptive depth: start at 2, try to increase if time allows
#         self.depth_limit = 2
#         while not self.is_time_exceeded():
#             for move in grid.getAvailableMoves():
#                 new_grid = grid.clone()
#                 new_grid.move(move[0])
#                 move_score = self.expectiminimax(new_grid, depth=0, is_maximizing=False, alpha=float('-inf'), beta=float('inf'))

#                 if move_score > best_score:
#                     best_score, best_move = move_score, move[0]

#             # Increase depth if time allows
#             if not self.is_time_exceeded():
#                 self.depth_limit += 1
#             else:
#                 break

#         # Return the best move, or a random move if none are available
#         return best_move if best_move is not None else random.choice(grid.getAvailableMoves())[0]

#     def expectiminimax(self, grid, depth, is_maximizing, alpha, beta):
#         """Runs expectiminimax with alpha-beta pruning to evaluate moves."""
#         if self.is_time_exceeded() or not grid.getAvailableMoves() or depth >= self.depth_limit:
#             return self.evaluate_grid(grid)
        
#         if is_maximizing:
#             max_score = float('-inf')
#             for move in grid.getAvailableMoves():
#                 new_grid = grid.clone()
#                 new_grid.move(move[0])
#                 score = self.expectiminimax(new_grid, depth + 1, False, alpha, beta)
#                 max_score = max(max_score, score)
#                 alpha = max(alpha, score)
#                 if beta <= alpha:
#                     break
#             return max_score
#         else:
#             min_score = 0
#             empty_cells = grid.getAvailableCells()
#             prob_2, prob_4 = 0.9, 0.1
            
#             for cell in empty_cells:
#                 for tile_value, probability in [(2, prob_2), (4, prob_4)]:
#                     new_grid = grid.clone()
#                     new_grid.insertTile(cell, tile_value)
#                     min_score += self.expectiminimax(new_grid, depth + 1, True, alpha, beta) * probability

#                 beta = min(beta, min_score)
#                 if beta <= alpha:
#                     break

#             return min_score / len(empty_cells) if empty_cells else 0

#     def evaluate_grid(self, grid):
#         """Evaluates the grid using optimized weighted heuristics."""
#         max_tile = self.get_max_tile(grid)
#         smoothness = self.calculate_smoothness(grid)
#         monotonicity = self.calculate_monotonicity(grid)
#         empty_cells = len(grid.getAvailableCells())
#         corner_bonus = self.corner_max_tile(grid)

#         # Adjusted heuristic weights for improved performance
#         return (
#             3.0 * math.log2(max_tile) +  # Log scaling for max tile
#             1.5 * smoothness +
#             1.5 * monotonicity +  # Higher weight for monotonicity
#             1.5 * empty_cells +   # Encourage open spaces to avoid board fill-up
#             2.0 * corner_bonus    # Encourages positioning of high tile in a corner
#         )

#     def is_time_exceeded(self):
#         """Checks if the move's time limit has been exceeded."""
#         return (time.process_time() - self.start_time) >= self.time_limit

#     def get_max_tile(self, grid):
#         """Returns the highest tile value on the grid."""
#         return max(max(row) for row in grid.map)

#     def calculate_smoothness(self, grid):
#         """Calculates smoothness score, favoring adjacent tiles with similar values."""
#         smoothness = 0
#         for x in range(4):
#             for y in range(4):
#                 if grid.map[x][y] != 0:
#                     value = grid.map[x][y]
#                     if x + 1 < 4 and grid.map[x + 1][y] != 0:
#                         smoothness -= abs(value - grid.map[x + 1][y])
#                     if y + 1 < 4 and grid.map[x][y + 1] != 0:
#                         smoothness -= abs(value - grid.map[x][y + 1])
#         return smoothness

#     def calculate_monotonicity(self, grid):
#         """Calculates a monotonicity score, favoring rows/columns with ordered values."""
#         monotonicity = 0
#         for row in grid.map:
#             for i in range(3):
#                 if row[i] < row[i + 1]:
#                     monotonicity += row[i + 1] - row[i]

#         for col in range(4):
#             for i in range(3):
#                 if grid.map[i][col] < grid.map[i + 1][col]:
#                     monotonicity += grid.map[i + 1][col] - grid.map[i][col]
#         return monotonicity

#     def corner_max_tile(self, grid):
#         """Adds a bonus if the max tile is in a corner, encouraging stable positioning."""
#         max_tile = self.get_max_tile(grid)
#         if grid.map[0][0] == max_tile or grid.map[0][3] == max_tile or grid.map[3][0] == max_tile or grid.map[3][3] == max_tile:
#             return max_tile
#         return 0



# import random
# import time
# import math
# from BaseAI import BaseAI

# class IntelligentAgent(BaseAI):
#     def __init__(self):
#         self.time_limit = 0.2  # Maximum time allowed per move
#         self.start_time = None
#         self.depth_limit = 2  # Starting depth limit, will increase adaptively

#     def getMove(self, grid):
#         """Selects the best move by running expectiminimax with alpha-beta pruning."""
#         self.start_time = time.process_time()
#         best_move, best_score = None, float('-inf')

#         # Adaptive depth: start at 2, try to increase if time allows
#         self.depth_limit = 2
#         while not self.is_time_exceeded():
#             for move in grid.getAvailableMoves():
#                 new_grid = grid.clone()
#                 new_grid.move(move[0])
#                 move_score = self.expectiminimax(new_grid, depth=0, is_maximizing=False, alpha=float('-inf'), beta=float('inf'))

#                 if move_score > best_score:
#                     best_score, best_move = move_score, move[0]

#             # Increase depth if time allows
#             if not self.is_time_exceeded():
#                 self.depth_limit += 1
#             else:
#                 break

#         # Return the best move, or a random move if none are available
#         return best_move if best_move is not None else random.choice(grid.getAvailableMoves())[0]

#     def expectiminimax(self, grid, depth, is_maximizing, alpha, beta):
#         """Runs expectiminimax with alpha-beta pruning to evaluate moves."""
#         if self.is_time_exceeded() or not grid.getAvailableMoves() or depth >= self.depth_limit:
#             return self.evaluate_grid(grid)
        
#         if is_maximizing:
#             max_score = float('-inf')
#             for move in grid.getAvailableMoves():
#                 new_grid = grid.clone()
#                 new_grid.move(move[0])
#                 score = self.expectiminimax(new_grid, depth + 1, False, alpha, beta)
#                 max_score = max(max_score, score)
#                 alpha = max(alpha, score)
#                 if beta <= alpha:
#                     break
#             return max_score
#         else:
#             min_score = 0
#             empty_cells = grid.getAvailableCells()
#             prob_2, prob_4 = 0.9, 0.1
            
#             for cell in empty_cells:
#                 for tile_value, probability in [(2, prob_2), (4, prob_4)]:
#                     new_grid = grid.clone()
#                     new_grid.insertTile(cell, tile_value)
#                     min_score += self.expectiminimax(new_grid, depth + 1, True, alpha, beta) * probability

#                 beta = min(beta, min_score)
#                 if beta <= alpha:
#                     break

#             return min_score / len(empty_cells) if empty_cells else 0

#     def evaluate_grid(self, grid):
#         """Evaluates the grid using optimized weighted heuristics."""
#         max_tile = self.get_max_tile(grid)
#         smoothness = self.calculate_smoothness(grid)
#         monotonicity = self.calculate_monotonicity(grid)
#         empty_cells = len(grid.getAvailableCells())
#         corner_bonus = self.corner_max_tile(grid)
#         opposite_corner_penalty = self.opposite_corner_low_tiles(grid)

#         # Adjusted heuristic weights for improved performance
#         return (
#             2.5 * math.log2(max_tile) +  # Log scaling for max tile
#             1.0 * smoothness +
#             5.0 * monotonicity +         # Higher weight for monotonicity
#             2.5 * empty_cells +          # Encourage open spaces to avoid board fill-up
#             2.0 * corner_bonus +         # Encourages positioning of high tile in a corner
#             1.0 * opposite_corner_penalty # Penalty for low tiles in the opposite corner
#         )

#     def is_time_exceeded(self):
#         """Checks if the time limit for the move has been exceeded."""
#         return (time.process_time() - self.start_time) >= self.time_limit

#     def get_max_tile(self, grid):
#         """Returns the highest tile value on the grid."""
#         return max(max(row) for row in grid.map)

#     def calculate_smoothness(self, grid):
#         """Calculates smoothness score, favoring adjacent tiles with similar values."""
#         smoothness = 0
#         for x in range(4):
#             for y in range(4):
#                 if grid.map[x][y] != 0:
#                     value = grid.map[x][y]
#                     if x + 1 < 4 and grid.map[x + 1][y] != 0:
#                         smoothness -= abs(value - grid.map[x + 1][y])
#                     if y + 1 < 4 and grid.map[x][y + 1] != 0:
#                         smoothness -= abs(value - grid.map[x][y + 1])
#         return smoothness

#     def calculate_monotonicity(self, grid):
#         """Calculates a monotonicity score, favoring rows/columns with ordered values."""
#         monotonicity = 0
#         for row in grid.map:
#             for i in range(3):
#                 if row[i] < row[i + 1]:
#                     monotonicity += row[i + 1] - row[i]

#         for col in range(4):
#             for i in range(3):
#                 if grid.map[i][col] < grid.map[i + 1][col]:
#                     monotonicity += grid.map[i + 1][col] - grid.map[i][col]
#         return monotonicity

#     def corner_max_tile(self, grid):
#         """Adds a bonus if the max tile is in a corner, encouraging stable positioning."""
#         max_tile = self.get_max_tile(grid)
#         if grid.map[0][0] == max_tile or grid.map[0][3] == max_tile or grid.map[3][0] == max_tile or grid.map[3][3] == max_tile:
#             return max_tile
#         return 0

#     def opposite_corner_low_tiles(self, grid):
#         """Encourages lower tiles in the corner opposite the max tile to avoid interference."""
#         max_tile_position = [(i, j) for i in range(4) for j in range(4) if grid.map[i][j] == self.get_max_tile(grid)]
#         if not max_tile_position:
#             return 0
        
#         max_tile_position = max_tile_position[0]
#         opposite_corner_value = 0

#         # Identify the opposite corner based on max tile's position
#         if max_tile_position == (0, 0):
#             opposite_corner_value = grid.map[3][3]
#         elif max_tile_position == (0, 3):
#             opposite_corner_value = grid.map[3][0]
#         elif max_tile_position == (3, 0):
#             opposite_corner_value = grid.map[0][3]
#         elif max_tile_position == (3, 3):
#             opposite_corner_value = grid.map[0][0]

#         # The lower the value in the opposite corner, the better the layout is for stability
#         return max(0, 2048 - opposite_corner_value)  # Encourages lower values in the opposite corner



# reached 2048
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


