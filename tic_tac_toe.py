import tkinter as tk

class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Tic-Tac-Toe")
        self.board = [' ' for _ in range(9)]
        self.buttons = [tk.Button(master, text=' ', font='normal 20 bold', height=4, width=7, command=lambda i=i: self.on_button_click(i)) for i in range(9)]
        self.create_board()

    def create_board(self):
        for i in range(9):
            self.buttons[i].grid(row=i//3, column=i%3)

    def on_button_click(self, index):
        if self.board[index] == ' ':
            self.board[index] = 'X'
            self.buttons[index].config(text='X')
            if self.check_winner('X'):
                self.end_game('You win!')
                return
            self.ai_move()
            if self.check_winner('O'):
                self.end_game('You lose!')
                return
            if ' ' not in self.board:
                self.end_game('Draw!')

    def ai_move(self):
        move = self.get_best_move()
        self.board[move] = 'O'
        self.buttons[move].config(text='O')

    def get_best_move(self):
        for i in range(9):
            if self.board[i] == ' ':
                return i  # Simple AI for demonstration

    def check_winner(self, player):
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]
        return any(all(self.board[pos] == player for pos in condition) for condition in win_conditions)

    def end_game(self, result):
        for button in self.buttons:
            button.config(state=tk.DISABLED)
        print(result)

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToeGUI(root)
    root.mainloop()
