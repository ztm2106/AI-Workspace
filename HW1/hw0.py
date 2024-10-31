import numpy as np
import heapq

# Helper function to calculate factorial
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Main function to return a string of first k factorials in reverse order
def reverse_factorials(k):
    factorials = []
    for i in range(1, k + 1):
        factorials.append(factorial(i))
    
    # Reverse the list and join as a string separated by commas
    return ','.join(map(str, factorials[::-1]))

# Example usage
k = 8
print(reverse_factorials(k))  # This should print: "40320,5040,720,120,24,6,2,1"


def descend_and_DeleteLast(y):
    # Sort y in descending order
    z = sorted(y, reverse=True)
    # Remove the last element from the sorted list
    z.pop()  # pop removes the last element
    return z

def reverse_list(x):
    # Reverse the list
    return x[::-1]

def concat_xy(x, y):
    # Assuming you want to concatenate corresponding elements from x and y into z
    z = []
    for i in range(min(len(x), len(y))):  # iterate over the smaller list
        z.append(x[i] + y[i])
    return z

def list_x_y(x, y):
    # Concatenate two lists
    z = x + y
    return z

x = [3,4,2,5,6,7,4]
y = [2,4,5,2,1,5,9]
print(" ")
print(descend_and_DeleteLast(y))
print(reverse_list(x))
print(concat_xy(x,y))
print(list_x_y(x,y))

def union_of_three_sets(set1, set2, set3):
    # Return the union of the three sets
    return set1.union(set2, set3)

def intersection_of_three_sets(set1, set2, set3):
    # Return the union of the three sets
    return set1.intersection(set2, set3)

def only_in_oneset(set1, set2, set3):
    only_in_set1 = set1 - set2 - set3
    only_in_set2 = set2 - set1 - set3
    only_in_set3 = set3 - set1 - set2
    
    # Return the union of elements unique to each set
    return only_in_set1 | only_in_set2 | only_in_set3


set1 = {1, 2, 3}
set2 = {3, 4, 5}
set3 = {5, 6, 7}
print(" ")
print(union_of_three_sets(set1, set2, set3))
print(intersection_of_three_sets(set1, set2, set3))
print(only_in_oneset(set1, set2, set3))


arr = np.array([[1,1,1,1,1],[1,0,0,0,1],[1,0,2,0,1],[1,0,0,0,1],[1,1,1,1,1]])

# def check_ones(arr):
#     x = (len(arr)/2) - 1
#     y = (len(arr)/2) - 1
#     element= []
    
#     check = [(x+1,y+2), (x-1,y+2), (x+2,y+1), (x+2,y-1), (x-1,y+2), (x-1,y-2), (x+1,y-2), (x-2,y+1), (x-2,y-1)]
#     for d in check:
#         element = element + check[d]
#     return element

def find_threatening_knights(board):
    # Define possible knight moves (row, column)
    knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), 
                    (1, 2), (1, -2), (-1, 2), (-1, -2)]
    
    # Find the position of the white pawn (2)
    white_pawn_pos = np.argwhere(board == 2)
    print(white_pawn_pos)
    if white_pawn_pos.size == 0:
        # No white pawn found on the board
        return []
    
    white_pawn_pos = tuple(white_pawn_pos[0])
    threats = []
    
    # Check for each possible knight move
    for move in knight_moves:
        knight_pos = (white_pawn_pos[0] + move[0], white_pawn_pos[1] + move[1])
        if (0 <= knight_pos[0] < board.shape[0] and 
            0 <= knight_pos[1] < board.shape[1] and 
            board[knight_pos[0], knight_pos[1]] == 1):
            threats.append(knight_pos)
    
    return threats

print(arr)
board = arr
print(find_threatening_knights(board))

def check_Empty(graph):
    i = 0
    for key in graph:
        if graph[key] == []:
            i = i+1
    
    return i

def check_Not_Empty(graph):
    i = 0
    for key in graph:
        if graph[key] != []:
            i = i+1
    
    return i

def check_unique(graph):
    arr = []  # This will hold the unique values
    for key in graph:
        for value in graph[key]:
            if value in arr:
                arr.remove(value)
            else:  
                arr.append(value)  
    
    return arr

def check_links(graph):
    links = []  # Initialize an empty list to store (key, value) pairs
    
    # Iterate through the graph
    for key in graph:
        if graph[key]:  # Check if the list of values for the current key is not empty
            for value in graph[key]:
                links.append((key, value))  # Append (key, value) tuple to the list
    
    # Convert the list of links to a NumPy array
    return np.array(links)



graph = {
"A": ["D", "E"],
"B": ["E", "F"],
"C": ["E"],
"D": ["A", "E"],
"E": ["A", "B", "C", "D"], "F": ["B"],
"G": []
}


print(" ")
print(check_Empty(graph))
print(check_Not_Empty(graph))
print(check_unique(graph))
print(check_links(graph))


class PriorityQueue:
    def __init__(self):
        self._queue = []  # This will store the heap
        self._counter = 0  # Counter to track the order of insertion
    
    def is_empty(self):
        # Return True if the queue is empty, False otherwise
        return len(self._queue) == 0
    
    def push(self, item, price):
        # Push an item onto the heap with priority based on price
        heapq.heappush(self._queue, (price, self._counter, item))
        self._counter += 1  # Increment counter to maintain insertion order
    
    def pop(self):
        # Pop the item with the highest priority (lowest price, then by order)
        if not self.is_empty():
            return heapq.heappop(self._queue)[2]  # Return only the item
        else:
            raise IndexError("pop from an empty priority queue")

pq = PriorityQueue()
pq = PriorityQueue() 
pq.push("orange",3.7) 
pq.push("carrot",5.1) 
pq.push("kiwi",2.4) 
pq.push("apple",3.7) 

while not pq.is_empty():
   print(pq.pop())


print(pq.pop())  # Output: "Banana" (lower price)
print(pq.pop())  # Output: "Apple" (same price as Orange but inserted first)
print(pq.pop())