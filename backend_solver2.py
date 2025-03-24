import heapq
from collections import deque
import time
import matplotlib.pyplot as plt


class Node:          #initialising the node
    def __init__(self, board, parent=None, move=None, cost=0, heuristic=0, depth=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic
        self.depth = depth

    def __eq__(self, other):
        return self.board == other.board

    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.board))


class Solver:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.size = len(initial_state)
        self.goal_positions = self.map_goal_positions(goal_state)

    def map_goal_positions(self, goal_state):
        size = len(goal_state)
        return {goal_state[row][col]: (row, col) for row in range(size) for col in range(size)}

    def find_neighbors(self, node): #getting all possible childs for each node
        neighbors = []
        blank_tile_pos = [(r, c) for r in range(self.size) for c in range(self.size) if node.board[r][c] == 0][0]
        row, col = blank_tile_pos
        potential_moves = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]

        for new_pos in potential_moves:
            if self.is_within_bounds(new_pos):
                new_board = self.swap_tiles(node.board, blank_tile_pos, new_pos)
                neighbors.append(Node(new_board, node, new_pos))

        return neighbors

    def is_within_bounds(self, position):  #checking if the position between 0 and 2 or not
        return 0 <= position[0] < self.size and 0 <= position[1] < self.size

    def swap_tiles(self, board, pos1, pos2):   #returnig a new board after swaping the zero with other tile
        new_board = [row[:] for row in board]
        new_board[pos1[0]][pos1[1]], new_board[pos2[0]][pos2[1]] = new_board[pos2[0]][pos2[1]], new_board[pos1[0]][pos1[1]]
        return new_board

    def calculate_heuristic(self, board, method=1): #calls either manhattan_distance or euclidean_distance
        if method == 1:
            return self.manhattan_distance(board)
        return self.euclidean_distance(board)

    def manhattan_distance(self, board):   #calculate the manhattan distance
        distance = 0
        for row in range(self.size):
            for col in range(self.size):
                tile = board[row][col]
                if tile != 0:
                    goal_row, goal_col = self.goal_positions[tile]
                    distance += abs(row - goal_row) + abs(col - goal_col)
        return distance

    def euclidean_distance(self, board):   #calculate the euclidean distance
        distance = 0
        for row in range(self.size):
            for col in range(self.size):
                tile = board[row][col]
                if tile != 0:
                    goal_row, goal_col = self.goal_positions[tile]
                    distance += ((row - goal_row) ** 2 + (col - goal_col) ** 2) ** 0.5
        return distance

    def is_solved(self, board):   #checking if the given node is the goal or not
        return board == self.goal_state

    def is_solvable(self):        # checking if the initial state can be solved or not
        flat_board = [tile for row in self.initial_state for tile in row if tile != 0]
        inversions = 0
        for i in range(len(flat_board)):
            for j in range(i + 1, len(flat_board)):
                if flat_board[i] > flat_board[j]:
                    inversions += 1
        return inversions % 2 == 0

    def bfs(self):               #initialising bfs code
        initial_node = Node(self.initial_state, depth=0)
        if self.is_solved(initial_node.board):
            return self.reconstruct_path(initial_node), 0, initial_node.depth

        frontier = deque([initial_node])
        explored = set()

        while frontier:
            current_node = frontier.popleft()
            if self.is_solved(current_node.board):
                return self.reconstruct_path(current_node), len(explored), current_node.depth

            neighbors = self.find_neighbors(current_node)
            for neighbor in neighbors:
                if neighbor not in explored:
                    neighbor.depth = current_node.depth + 1
                    frontier.append(neighbor)
                    explored.add(neighbor)

        return None, 0, 0

    def dfs(self):             #initialising dfs code
        initial_node = Node(self.initial_state, depth=0)  # Initialize depth for the starting node
        if self.is_solved(initial_node.board):
            return self.reconstruct_path(initial_node), 0, 0  # Return path, nodes explored, max depth

        frontier = [initial_node]
        explored = set()
        max_depth = 0

        while frontier:
            current_node = frontier.pop()
            explored.add(current_node)

            if self.is_solved(current_node.board):
                return self.reconstruct_path(current_node), len(explored), max_depth

            max_depth = max(max_depth, current_node.depth)

            neighbors = self.find_neighbors(current_node)
            for neighbor in neighbors:
                if neighbor not in explored:
                    neighbor.depth = current_node.depth + 1
                    frontier.append(neighbor)

        return None, 0, max_depth

    def iddfs(self):         #initialising iddfs code
        depth = 0
        total_explored = 0
        while True:
            result, explored_nodes = self.dls(self.initial_state, self.goal_state, depth)
            total_explored += explored_nodes
            if result is not None:
                return result, total_explored ,depth
            depth += 1

    def dls(self, initial_state, goal_state, depth):    #initialising dls code which is called from iddfs function
        initial_node = Node(initial_state)
        if self.is_solved(initial_node.board):
            return self.reconstruct_path(initial_node), 1

        if depth == 0:
            return None, 1

        explored_count = 0
        frontier = [(initial_node, depth)]
        explored = set()

        while frontier:
            current_node, current_depth = frontier.pop()
            explored_count += 1

            if self.is_solved(current_node.board):
                return self.reconstruct_path(current_node), explored_count

            if current_node not in explored:
                explored.add(current_node)

                if current_depth > 0:
                    neighbors = self.find_neighbors(current_node)
                    for neighbor in neighbors:
                        frontier.append((neighbor, current_depth - 1))

        return None, explored_count

    def a_star(self):
        return self.a_star_helper(method=1)

    def a_star_euclidean(self):
        return self.a_star_helper(method=2)

    def a_star_helper(self, method):
        initial_node = Node(self.initial_state, depth=0, heuristic=self.calculate_heuristic(self.initial_state, method))

        frontier = []
        heapq.heappush(frontier, initial_node)
        explored = set()

        while frontier:
            current_node = heapq.heappop(frontier)

            if self.is_solved(current_node.board):
                return self.reconstruct_path(current_node), len(explored), current_node.depth

            explored.add(current_node)

            for neighbor in self.find_neighbors(current_node):
                neighbor.cost = current_node.cost + 1
                neighbor.depth = current_node.depth + 1
                neighbor.heuristic = self.calculate_heuristic(neighbor.board, method)
                neighbor.total_cost = neighbor.cost + neighbor.heuristic

                if neighbor not in explored and neighbor not in frontier:
                    heapq.heappush(frontier, neighbor)

        return None, 0, 0

    def reconstruct_path(self, node):       #getting the paths of the solution in the right way
        stack = []
        while node:
            stack.append(node.board)
            node = node.parent
        path = []
        while stack:
            path.append(stack.pop())
        return path

    def print_solution_steps(self, solution):    #printing the solution steps
        for step_num, board in enumerate(solution):
            step_text = f"Step {step_num}:\n"
            for row in board:
                step_text += ' '.join(str(tile) if tile != 0 else ' ' for tile in row) + '\n'
            print(step_text)

    def solve_with_timing(self):        # getting the result stats to be graphed
        results = []
        nodes_explored_list = []
        steps_list = []
        methods = [self.bfs, self.dfs, self.iddfs, self.a_star, self.a_star_euclidean]
        method_names = ["BFS", "DFS", "IDDFS", "A*", "A* (Euclidean)"]
        times = []
        depth_list = []

        for method, method_name in zip(methods, method_names):
            start_time = time.time()
            solution, nodes_explored , depth = method()
            time_taken = time.time() - start_time

            if solution:
                steps = len(solution) - 1
                self.print_solution_steps(solution)
                results.append((method_name, time_taken, nodes_explored, steps , depth))
                times.append(time_taken)
                nodes_explored_list.append(nodes_explored)
                steps_list.append(steps)
                depth_list.append(depth)

        return results, times, nodes_explored_list, steps_list , depth_list


    def plot_histograms(self, results, times):    # making the required graphs
        plt.figure(figsize=(16, 8))
        plt.subplot(2, 2, 1)
        plt.bar([method for method, _, _, _, _ in results], times, color='blue')
        plt.xlabel('Method')
        plt.ylabel('Time (seconds)')
        plt.title('Time taken by each solving method')
        plt.yscale('log')

        plt.subplot(2, 2, 2)
        plt.bar([method for method, _, _, _, _ in results], [nodes for _, _, nodes, _, _ in results],
                color='orange')
        plt.xlabel('Method')
        plt.ylabel('Nodes Explored')
        plt.title('Number of Nodes Explored by each solving method')

        plt.subplot(2, 2, 3)
        plt.bar([method for method, _, _, steps, _ in results], [steps for _, _, _, steps, _ in results],
                color='green')
        plt.xlabel('Method')
        plt.ylabel('Steps')
        plt.title('Number of Steps Taken by each solving method')

        plt.subplot(2, 2, 4)
        plt.bar([method for method, _, _, _, depth in results], [depth for _, _, _, _, depth in results],
                color='red')
        plt.xlabel('Method')
        plt.ylabel('Depth')
        plt.title('Maximum Depth Reached by each solving method')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # initial_state = [
    #     [1, 2, 3],
    #     [5, 6, 0],
    #     [7, 8, 4]
    # ]

    initial_state = [
        [1, 2, 3],
        [4, 0, 5],
        [7, 8, 6]
    ]
    # initial_state = [
    #     [1, 4, 2],
    #     [3, 5, 8],
    #     [0, 6, 7]
    # ]

    goal_state = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]

    solver = Solver(initial_state, goal_state)

    if solver.is_solvable():
        print("Puzzle is solvable. Attempting to solve...")
        results, times, nodes_explored, steps, depths = solver.solve_with_timing()
        for (method, time_taken, nodes, steps, depths) in results:
            print(f"{method} Solution: Nodes explored: {nodes}, Time taken: {time_taken:.6f} seconds, Steps: {steps}, depth: {depths}")

        solver.plot_histograms(results, times)
    else:
        print("Puzzle is not solvable.")
