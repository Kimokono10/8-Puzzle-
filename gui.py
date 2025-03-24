import tkinter as tk
from tkinter import messagebox
import time
from random import sample
from backend_solver2 import Solver


class PuzzleGUI:
    def __init__(self, root, solver):
        self.root = root
        self.root.title("8 Puzzle Solver")

        self.solver = solver
        self.goal_state = self.solver.goal_state

        self.board = self.solver.initial_state
        self.buttons = []
        for m in range(3):
            row = []
            for n in range(3):
                button = tk.Button(self.root, text=str(self.board[m][n]) if self.board[m][n] != 0 else "",
                                   width=10, height=4, font=('Arial', 18),
                                   command=lambda i=m, j=n: self.on_tile_click(i, j))
                button.grid(row=m, column=n + 1)
                row.append(button)
            self.buttons.append(row)

        self.bfs_button = tk.Button(self.root, text="Solve with BFS", command=self.solve_with_bfs)
        self.bfs_button.grid(row=3, column=0)

        self.dfs_button = tk.Button(self.root, text="Solve with DFS", command=self.solve_with_dfs)
        self.dfs_button.grid(row=3, column=1)

        self.iddfs_button = tk.Button(self.root, text="Solve with IDDFS", command=self.solve_with_iddfs)
        self.iddfs_button.grid(row=3, column=2)

        self.astar_button = tk.Button(self.root, text="Solve with A* (Manhattan)", command=self.solve_with_astar)
        self.astar_button.grid(row=3, column=3)

        self.astar_euclidean_button = tk.Button(self.root, text="Solve with A* (Euclidean)",
                                                command=self.solve_with_astar_euclidean)
        self.astar_euclidean_button.grid(row=3, column=4)

        self.randomize_button = tk.Button(self.root, text="Randomize", command=self.randomize_board)
        self.randomize_button.grid(row=4, column=2)

    def update_board(self):
        for i in range(3):
            for j in range(3):
                tile = self.board[i][j]
                self.buttons[i][j].config(text=str(tile) if tile != 0 else "")

    def on_tile_click(self, i, j):
        if i != 2 and self.board[i + 1][j] == 0:
            self.board[i][j], self.board[i + 1][j] = self.board[i + 1][j], self.board[i][j]
        if i != 0 and self.board[i - 1][j] == 0:
            self.board[i][j], self.board[i - 1][j] = self.board[i - 1][j], self.board[i][j]
        if j != 2 and self.board[i][j + 1] == 0:
            self.board[i][j], self.board[i][j + 1] = self.board[i][j + 1], self.board[i][j]
        if j != 0 and self.board[i][j - 1] == 0:
            self.board[i][j], self.board[i][j - 1] = self.board[i][j - 1], self.board[i][j]
        self.update_board()
        if self.solver.is_solved(self.board):
            messagebox.showinfo("Congrats", "Solution found!")

    def solve_with_bfs(self):
        solution, nodes_explored ,depth = self.solver.bfs()
        if solution:
            steps = len(solution) - 1
            messagebox.showinfo("Solution", f"Solved with BFS in {nodes_explored} nodes explored, solved in {steps} steps")
            self.print_solution_steps(solution)
            self.animate_solution(solution)
        else:
            messagebox.showerror("Error", "No solution found with BFS.")

    def solve_with_iddfs(self):
        solution, nodes_explored ,depth = self.solver.iddfs()
        if solution:
            steps = len(solution) - 1
            messagebox.showinfo("Solution", f"Solved with IDDFS in {nodes_explored} nodes explored, solved in {steps} steps")
            self.print_solution_steps(solution)
            self.animate_solution(solution)
        else:
            messagebox.showerror("Error", "No solution found with IDDFS.")

    def solve_with_dfs(self):
        solution, nodes_explored ,depth = self.solver.dfs()
        if solution:
            steps = len(solution) - 1
            messagebox.showinfo("Solution", f"Solved with DFS in {nodes_explored} nodes explored, solved in {steps} steps")
            self.print_solution_steps(solution)
            self.animate_solution(solution)
        else:
            messagebox.showerror("Error", "No solution found with DFS.")

    def solve_with_astar(self):
        solution, nodes_explored ,depth = self.solver.a_star()
        if solution:
            steps = len(solution) - 1
            messagebox.showinfo("Solution", f"Solved with A* (Manhattan) in {nodes_explored} nodes explored, solved in {steps} steps")
            self.print_solution_steps(solution)
            self.animate_solution(solution)
        else:
            messagebox.showerror("Error", "No solution found with A* (Manhattan).")

    def solve_with_astar_euclidean(self):
        solution, nodes_explored ,depth = self.solver.a_star_euclidean()
        if solution:
            steps = len(solution) - 1
            messagebox.showinfo("Solution", f"Solved with A* (Euclidean) in {nodes_explored} nodes explored, solved in {steps} steps")
            self.print_solution_steps(solution)
            self.animate_solution(solution)
        else:
            messagebox.showerror("Error", "No solution found with A* (Euclidean).")

    def randomize_board(self):
        self.board = self.generate_random_state()
        self.solver = Solver(self.board, self.goal_state)
        self.update_board()

    def generate_random_state(self):
        while True:
            board = sample(range(9), 9)
            board = [board[i:i + 3] for i in range(0, 9, 3)]
            if Solver(board, self.goal_state).is_solvable():
                return board

    def animate_solution(self, solution):
        for step in solution:
            self.board = step
            self.update_board()
            self.root.update()
            time.sleep(0.35)
        if self.solver.is_solved(self.board):
            messagebox.showinfo("Congrats", "Solution found!")

    def print_solution_steps(self, solution):
        for step_num, board in enumerate(solution):
            step_text = f"Step {step_num}:\n"
            for row in board:
                step_text += ' '.join(str(tile) if tile != 0 else ' ' for tile in row) + '\n'
            print(step_text)


if __name__ == "__main__":

    initial_state = [
        [1, 4, 2],
        [3, 5, 8],
        [0, 6, 7]
    ]
    # initial_state = [
    #       [3, 1, 2],
    #       [0, 4, 5],
    #        [6, 7, 8]
    #   ]
    # initial_state = [
    #     [1, 2, 3],
    #     [5, 6, 0],
    #     [7, 8, 4]
    # ]

    # initial_state = [
    #     [1, 2, 3],
    #     [4, 0, 5],
    #     [7, 8, 6]
    # ]


    #initial_state = [
   #     [4, 1, 2],
  #      [3, 5, 8],
  #      [0, 6, 7]
  #  ]
    goal_state = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]



    solver = Solver(initial_state, goal_state)
    if solver.is_solvable():
        print("Puzzle is solvable. Attempting to solve...")
        root = tk.Tk()
        app = PuzzleGUI(root, solver)
        root.mainloop()
    else:
        messagebox.showerror("Error", "Puzzle is not solvable.")
