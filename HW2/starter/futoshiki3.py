"""
Each futoshiki board is represented as a dictionary with string keys and int values.
e.g. my_board['A1'] = 8

Empty values in the board are represented by 0

An * after the letter indicates the inequality between the row represented
by the letter and the next row.
e.g. my_board['A*1'] = '<' 
means the value at A1 must be less than the value at B1

Similarly, an * after the number indicates the inequality between the
column represented by the number and the next column.
e.g. my_board['A1*'] = '>' 
means the value at A1 is greater than the value at A2

Empty inequalities in the board are represented as '-'
"""
import sys

#======================================================================#
#*#*#*# Optional: Import any allowed libraries you may need here #*#*#*#
#======================================================================#

import numpy as np
import time

#=================================#
#*#*#*# Your code ends here #*#*#*#
#=================================#

ROW = "ABCDEFGHI"
COL = "123456789"

class Board:
    '''
    Class to represent a board, including its configuration, dimensions, and domains
    '''
    
    def get_board_dim(self, str_len):
        '''
        Returns the side length of the board given a particular input string length
        '''
        d = 4 + 12 * str_len
        n = (2+np.sqrt(4+12*str_len))/6
        if(int(n) != n):
            raise Exception("Invalid configuration string length")
        
        return int(n)
        
    def get_config_str(self):
        '''
        Returns the configuration string
        '''
        return self.config_str
        
    def get_config(self):
        '''
        Returns the configuration dictionary
        '''
        return self.config
        
    def get_variables(self):
        '''
        Returns a list containing the names of all variables in the futoshiki board
        '''
        variables = []
        for i in range(0, self.n):
            for j in range(0, self.n):
                variables.append(ROW[i] + COL[j])
        return variables
    
    def convert_string_to_dict(self, config_string):
        '''
        Parses an input configuration string, retuns a dictionary to represent the board configuration
        as described above
        '''
        config_dict = {}
        
        for i in range(0, self.n):
            for j in range(0, self.n):
                cur = config_string[0]
                config_string = config_string[1:]
                
                config_dict[ROW[i] + COL[j]] = int(cur)
                
                if(j != self.n - 1):
                    cur = config_string[0]
                    config_string = config_string[1:]
                    config_dict[ROW[i] + COL[j] + '*'] = cur
                    
            if(i != self.n - 1):
                for j in range(0, self.n):
                    cur = config_string[0]
                    config_string = config_string[1:]
                    config_dict[ROW[i] + '*' + COL[j]] = cur
                    
        return config_dict
        
    def print_board(self):
        '''
        Prints the current board to stdout
        '''
        config_dict = self.config
        for i in range(0, self.n):
            for j in range(0, self.n):
                cur = config_dict[ROW[i] + COL[j]]
                if(cur == 0):
                    print('_', end=' ')
                else:
                    print(str(cur), end=' ')
                
                if(j != self.n - 1):
                    cur = config_dict[ROW[i] + COL[j] + '*']
                    if(cur == '-'):
                        print(' ', end=' ')
                    else:
                        print(cur, end=' ')
            print('')
            if(i != self.n - 1):
                for j in range(0, self.n):
                    cur = config_dict[ROW[i] + '*' + COL[j]]
                    if(cur == '-'):
                        print(' ', end='   ')
                    else:
                        print(cur, end='   ')
            print('')
    
    def __init__(self, config_string):
        '''
        Initialising the board
        '''
        self.config_str = config_string
        self.n = self.get_board_dim(len(config_string))
        if(self.n > 9):
            raise Exception("Board too big")
            
        self.config = self.convert_string_to_dict(config_string)
        self.domains = self.reset_domains()
        
        self.forward_checking(self.get_variables())
        
        
    def __str__(self):
        '''
        Returns a string displaying the board in a visual format. Same format as print_board()
        '''
        output = ''
        config_dict = self.config
        for i in range(0, self.n):
            for j in range(0, self.n):
                cur = config_dict[ROW[i] + COL[j]]
                if(cur == 0):
                    output += '_ '
                else:
                    output += str(cur)+ ' '
                
                if(j != self.n - 1):
                    cur = config_dict[ROW[i] + COL[j] + '*']
                    if(cur == '-'):
                        output += '  '
                    else:
                        output += cur + ' '
            output += '\n'
            if(i != self.n - 1):
                for j in range(0, self.n):
                    cur = config_dict[ROW[i] + '*' + COL[j]]
                    if(cur == '-'):
                        output += '    '
                    else:
                        output += cur + '   '
            output += '\n'
        return output
        
    def reset_domains(self):
        '''
        Resets the domains of the board assuming no enforcement of constraints
        '''
        domains = {}
        variables = self.get_variables()
        for var in variables:
            if(self.config[var] == 0):
                domains[var] = [i for i in range(1,self.n+1)]
            else:
                domains[var] = [self.config[var]]
                
        self.domains = domains
                
        return domains
    
    def update_config_str(self):
        '''
        Update self.config_str by converting the current board configuration (self.config)
        back to the string format.
        '''
        result = []
        
        for i in range(self.n):
            for j in range(self.n):
                result.append(str(self.config[ROW[i] + COL[j]]))
                if j != self.n - 1:
                    result.append(self.config[ROW[i] + COL[j] + '*'])  # Horizontal inequality
            if i != self.n - 1:
                for j in range(self.n):
                    result.append(self.config[ROW[i] + '*' + COL[j]])  # Vertical inequality

        self.config_str = ''.join(result)
        
    def forward_checking(self, reassigned_variables):
        '''
        Runs the forward checking algorithm to restrict the domains of all variables based on the values
        of reassigned variables
        '''
        #======================================================================#
        #*#*#*# TODO: Write your implementation of forward checking here #*#*#*#
        #======================================================================#

        # Initialize a queue with the variables whose assignments have been modified
        queue = reassigned_variables

        # Continue the process until there are no more variables to check
        while queue:
            # Remove the first variable from the queue to process it
            var = queue.pop(0)
            assignment = self.config[var]  # Current value of the variable in the configuration
            
            # Skip if the variable has not been assigned a value
            if assignment == 0:
                continue

            # Retrieve all neighbors of the variable (in both row and column)
            neighbors = self.get_neighbors(var)
            all_neighbors = neighbors['row'] + neighbors['col']  # Combine row and column neighbors

            # Iterate through each neighboring variable to update their domains
            for neighbor in all_neighbors:
                # If the assigned value exists in the neighbor's domain, remove it
                if assignment in self.domains[neighbor]:
                    self.domains[neighbor].remove(assignment)
                    
                    # If the neighbor's domain is now empty, a constraint is violated
                    if not self.domains[neighbor]:
                        return False
                    
                    # If only one value is left in the neighbor's domain, assign it to the configuration
                    if len(self.domains[neighbor]) == 1:
                        self.config[neighbor] = self.domains[neighbor][0]
                        queue.append(neighbor)  # Add the neighbor to the queue to propagate the change

            # Ensure that the current assignment still satisfies all constraints for this variable
            if not self.check_constraints(var, assignment, neighbors):
                return False  # Return False if any constraint is violated

        # Return True if the algorithm completes without conflicts
        return True

        #=================================#
        #*#*#*# Your code ends here #*#*#*#
        #=================================#
        
    #=================================================================================#
    #*#*#*# Optional: Write any other functions you may need in the Board Class #*#*#*#
    #=================================================================================#
    # check for constraint violations
    def check_constraints(self, var, value, neighbors):
        # Determine the row and column indices for the variable in the grid
        currRow = ROW.index(var[0])
        currCol = COL.index(var[1])

        # Iterate through each directional constraint in the neighbors dictionary
        for rowChange, colChange, key, expr in neighbors['directions']:
            # Calculate the neighbor's position by applying row and column changes
            i, j = currRow + rowChange, currCol + colChange
            
            # Check if the neighbor's position is within the grid boundaries
            if 0 <= i < self.n and 0 <= j < self.n:
                
                # Construct the neighbor's variable identifier
                neighbor = ROW[i] + COL[j]
                
                # Determine the constraint symbol key (direct or calculated based on position)
                key = key if isinstance(key, str) else key(i, j)
                
                # Retrieve the constraint symbol (like '<' or '>') from the configuration
                sign = self.config.get(key, '-')
                
                # Get the current value assigned to the neighbor variable
                neighbor_value = self.config[neighbor]

                # Check for any directional constraints if the neighbor has a non-zero assignment
                if sign != '-' and neighbor_value != 0:
                    # Evaluate if the constraint (e.g., '<' or '>') holds between the values
                    if (sign == '<' and not expr(value, neighbor_value)) or \
                       (sign == '>' and not expr(neighbor_value, value)):
                        return False  # Return False if a constraint is violated

        # Return True if all constraints are satisfied
        return True
    

    

    def get_neighbors(self, currentCell):
        # Determine the indices for the row and column of the current cell
        row = ROW.index(currentCell[0])
        col = COL.index(currentCell[1])

        # Initialize lists to hold neighboring cells in the same row and column
        same_row, same_col = [], []

        # Iterate over the grid size to find neighbors in the same row and column
        for i in range(self.n):
            # Identify cell in the same column at the ith row
            col_neighbor = ROW[i] + COL[col]
            if col_neighbor != currentCell:  # Exclude the current cell itself
                same_col.append(col_neighbor)
            
            # Identify cell in the same row at the ith column
            row_neighbor = ROW[row] + COL[i]
            if row_neighbor != currentCell:  # Exclude the current cell itself
                same_row.append(row_neighbor)

        # Define neighbors with directional constraints
        neighbors = {
            'row': same_row,  # Cells in the same row as currentCell
            'col': same_col,  # Cells in the same column as currentCell
            'directions': [
                # Right neighbor with constraint that currentCell's value < right neighbor's value
                (0, 1, currentCell + '*', lambda value, neighbor: value < neighbor),
                
                # Left neighbor with constraint that left neighbor's value < currentCell's value
                (0, -1, lambda letter, num: ROW[letter] + COL[num] + '*', lambda value, neighbor: neighbor < value),
                
                # Below neighbor with constraint that currentCell's value < below neighbor's value
                (1, 0, currentCell[0] + '*' + currentCell[1], lambda value, neighbor: value < neighbor),
                
                # Above neighbor with constraint that above neighbor's value < currentCell's value
                (-1, 0, lambda letter, num: ROW[letter] + '*' + COL[num], lambda value, neighbor: neighbor < value)
            ]
        }
        
        return neighbors  # Return dictionary of neighbors by type and direction


    def is_consistent(self, var, value):
        # Retrieve the neighbors of the variable (cell) in question
        neighbors = self.get_neighbors(var)

        # Check for conflicts with values in the same row
        for neighbor in neighbors['row']:
            # If any neighbor in the row has the same value, return False (inconsistent)
            if self.config[neighbor] == value:
                return False
            
        # Check for conflicts with values in the same column
        for neighbor in neighbors['col']:
            # If any neighbor in the column has the same value, return False (inconsistent)
            if self.config[neighbor] == value:
                return False

        # Check for conflicts with directional constraints (e.g., inequalities) using neighbors
        if not self.check_constraints(var, value, neighbors):
            return False  # Return False if directional constraints are violated

        # If all checks pass, return True (assignment is consistent)
        return True


    #=================================#
    #*#*#*# Your code ends here #*#*#*#
    #=================================#

#================================================================================#
#*#*#*# Optional: You may write helper functions in this space if required #*#*#*#
#================================================================================#        

#=================================#
#*#*#*# Your code ends here #*#*#*#
#=================================#

def backtracking(board):
    '''
    Performs the backtracking algorithm to solve the board
    Returns only a solved board
    '''
    #==========================================================#
    #*#*#*# TODO: Write your backtracking algorithm here #*#*#*#
    #==========================================================#

    # Base case: if all variables have been assigned a non-zero value, the board is solved
    if all(board.config[v] != 0 for v in board.get_variables()):
        # Assign final values from domains to configuration
        for v in board.get_variables():
            board.config[v] = board.domains[v][0]
        return board

    # List to store variables that are not yet fully assigned (domain has >1 possible value)
    unassigned = []
    for v in board.get_variables():
        if len(board.domains[v]) > 1:  # Multiple possible values in domain
            unassigned.append(v)
    
    # If no unassigned variables remain but the board isn't fully solved, return None (backtrack)
    if not unassigned:
        return None

    # Minimum Remaining Values (MRV) heuristic: choose the variable with the smallest domain
    domain_sizes_dict = {v: len(board.domains[v]) for v in unassigned}
    mrv_var = min(domain_sizes_dict, key=domain_sizes_dict.get)  # Variable with smallest domain

    # Iterate over each possible value in the selected variable's domain
    for value in board.domains[mrv_var]:
        # Check if assigning this value maintains consistency
        if board.is_consistent(mrv_var, value):

            # Save the current state of the domains and configuration for potential backtracking
            domain_restore = {cell: list(domains) for cell, domains in board.domains.items()}
            config_restore = dict(board.config)

            # Assign the value to the configuration and update domain to only contain this value
            board.config[mrv_var] = value
            board.domains[mrv_var] = [value]

            # Perform forward checking to eliminate inconsistent values from neighbors' domains
            if board.forward_checking([mrv_var]):
                # Recursively attempt to solve with this partial assignment
                result = backtracking(board)
                if result is not None:  # If a solution is found, return it
                    return result
            
            # Restore the previous configuration and domain state (undo assignment)
            board.domains = domain_restore
            board.config = config_restore

    # If no valid assignment is found, return None to backtrack further
    return None

    #=================================#
    #*#*#*# Your code ends here #*#*#*#
    #=================================#
    
def solve_board(board):
    '''
    Runs the backtrack helper and times its performance.
    Returns the solved board and the runtime
    '''
    #================================================================#
    #*#*#*# TODO: Call your backtracking algorithm and time it #*#*#*#
    #================================================================#
    start_time = time.time()
    solved_board = backtracking(board)
    end_time = time.time()
    runtime = end_time - start_time

    solved_board.update_config_str()  # Update config_str for the final solved board

    with open('output.txt', 'a') as outfile:
        if isinstance(solved_board, Board):
            outfile.write(solved_board.get_config_str() + '\n')
        else:
            raise ValueError("Solved board is not a Board object.")

    return solved_board, runtime

    #=================================#
    #*#*#*# Your code ends here #*#*#*#
    #=================================#

def print_stats(runtimes):
    '''
    Prints a statistical summary of the runtimes of all the boards
    '''
    min = 100000000000
    max = 0
    sum = 0
    n = len(runtimes)

    for runtime in runtimes:
        sum += runtime
        if(runtime < min):
            min = runtime
        if(runtime > max):
            max = runtime

    mean = sum/n

    sum_diff_squared = 0

    for runtime in runtimes:
        sum_diff_squared += (runtime-mean)*(runtime-mean)

    std_dev = np.sqrt(sum_diff_squared/n)

    print("\nRuntime Statistics:")
    print("Number of Boards = {:d}".format(n))
    print("Min Runtime = {:.8f}".format(min))
    print("Max Runtime = {:.8f}".format(max))
    print("Mean Runtime = {:.8f}".format(mean))
    print("Standard Deviation of Runtime = {:.8f}".format(std_dev))
    print("Total Runtime = {:.8f}".format(sum))

if __name__ == '__main__':
    if len(sys.argv) > 1:

        # Running futoshiki solver with one board $python3 futoshiki.py <input_string>.
        print("\nInput String:")
        print(sys.argv[1])
        
        print("\nFormatted Input Board:")
        board = Board(sys.argv[1])
        board.print_board()
        
        solved_board, runtime = solve_board(board)
        
        print("\nSolved String:")
        print(solved_board.get_config_str())
        
        print("\nFormatted Solved Board:")
        solved_board.print_board()
        
        print_stats([runtime])

        # Write board to file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        outfile.write(solved_board.get_config_str())
        outfile.write('\n')
        outfile.close()

    else:
        # Running futoshiki solver for boards in futoshiki_start.txt $python3 futoshiki.py

        #  Read boards from source.
        src_filename = 'futoshiki_start.txt'
        try:
            srcfile = open(src_filename, "r")
            futoshiki_list = srcfile.read()
            srcfile.close()
        except:
            print("Error reading the sudoku file %s" % src_filename)
            exit()

        # Setup output file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        
        runtimes = []

        # Solve each board using backtracking
        for line in futoshiki_list.split("\n"):
            
            print("\nInput String:")
            print(line)
            
            print("\nFormatted Input Board:")
            board = Board(line)
            board.print_board()
            
            solved_board, runtime = solve_board(board)
            runtimes.append(runtime)
            
            print("\nSolved String:")
            print(solved_board.get_config_str())
            
            print("\nFormatted Solved Board:")
            solved_board.print_board()

            # Write board to file
            outfile.write(solved_board.get_config_str())
            outfile.write('\n')

        # Timing Runs
        print_stats(runtimes)
        
        outfile.close()
        print("\nFinished all boards in file.\n")