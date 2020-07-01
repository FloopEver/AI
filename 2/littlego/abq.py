import pickle
import numpy as np
import os
import random
import sys
from copy import deepcopy

from read import readInput
from write import writeOutput
from host import GO

WIN_REWARD = 1.0
DRAW_REWARD = 0.5
LOSS_REWARD = -0.2

class QLearner():
    """
    piece_type: 1('X') or 2('O')
    """

    def __init__(self, alpha = 0.7, gamma = 0.9, initial_value = 0.5, side = None):
        if not (0 < gamma <= 1):
            raise ValueError("Invalid gamma value : %s" % gamma)
        self.side = side
        self.alpha = alpha
        self.gamma = gamma
        self.X_qvalues = {}
        self.O_qvalues = {}
        self.history_status = []
        self.initial_value = initial_value

    def read(self):
        """
        At the beginning of every turn, read qvalues from txt.
        If this is not the first move(history_status exists), then read history_status.
        """
        if (self.side == 1) and (os.path.exists("X_qvalues.txt")):
            file = open("X_qvalues.txt", "rb")
            self.X_qvalues = pickle.load(file)
            file.close()
        elif (self.side == 2) and (os.path.exists("O_qvalues.txt")):
            file = open("O_qvalues.txt", "rb")
            self.O_qvalues = pickle.load(file)
            file.close()
        if os.path.exists("history_status.txt"):
            file = open("history_status.txt", "rb")
            self.history_status = pickle.load(file)
            file.close()
        # print (self.X_qvalues, self.O_qvalues, self.history_status)
        return

    def save(self, end_flag):
        """
        Save history_status after every move.
        Save qvalues after the game ended.
        """
        file = open("history_status.txt", "wb+")
        pickle.dump(self.history_status, file)
        file.close()
        if end_flag == 1:
            if self.X_qvalues != {}:
                file = open("X_qvalues.txt", "wb+")
                pickle.dump(self.X_qvalues, file)
                file.close()
            elif self.O_qvalues != {}:
                file = open("O_qvalues.txt", "wb+")
                pickle.dump(self.O_qvalues, file)
                file.close()
            os.remove("history_status.txt")
        return

    def findq(self, state): 
        """
        Find curr_qvalues for current state.
        If no curr_qvalues, initiate it with self.initial_value(0.5)
        Return curr_qvalues
        """
        if self.side == 1:
            side_qvalues = self.X_qvalues
        else:
            side_qvalues = self.O_qvalues
        if state not in side_qvalues:
            initial_q = np.zeros((5, 5))
            initial_q.fill(self.initial_value)
            side_qvalues[state] = initial_q
        return side_qvalues[state]

    def _select_best_move(self, curr_state, possible_placements): 
        """
        Find the max possible move in curr_qvalues.
        If max is not unique, use minmax to find the best move.
        OW, max is the best move.
        Return best move.
        """
        curr_qvalues = self.findq(curr_state)
        max_values = self._find_max(possible_placements, curr_qvalues)
        if len(max_values) != 0:
            re_move = max_values[0]
        else:
            state = []
            for i in range(0, 25, 5):
                state.append([int(x) for x in curr_state[i: i+5]])
            re_move = self.alphabeta(state, 24, -np.inf, np.inf, self.side)[1]
        row = re_move[0]
        col = re_move[1]
        return row, col

    def alphabeta(self, state, depth, a, b, side):
        go = GO(5)       
        go.set_board(side, state, state)
        go.remove_died_pieces(side)
        state = go.board

        # Reach the max depth
        if depth == 0:
            return (go.score(2) + 3 - go.score(1), (1,2))

        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, self.side, test_check = True):
                    possible_placements.append((i,j))

        # No possible move:         
        if possible_placements == []:
            return (go.score(2) + 3 - go.score(1), (1,2))

        str_state = ''.join([str(state[i][j]) for i in range(N) for j in range(N)])
        curr_qvalues = self.findq(str_state)
        max_moves = self._find_max(possible_placements, curr_qvalues) 

        # Choose Max. (Side = 2)
        if side == 2:
            next_side = 1
            re_move = max_moves[0]
            for next_move in max_moves:
                next_state = deepcopy(state)
                next_state[next_move[0]][next_move[1]] = next_side
                if a < self.alphabeta(next_state, depth-1, a, b, next_side)[0]:
                    a = self.alphabeta(next_state, depth-1, a, b, next_side)[0]
                    re_move = next_move   
                if b <= a:
                    break                       
            return (a, re_move)

        # Choose Min. (Side = 1)
        else:
            next_side = 2
            re_move = max_moves[0]
            for next_move in max_moves:
                next_state = deepcopy(state)
                next_state[next_move[0]][next_move[1]] = next_side
                if b > self.alphabeta(next_state, depth-1, a, b, next_side)[0]:
                    b = self.alphabeta(next_state, depth-1, a, b, next_side)[0]
                    re_move = next_move
                if b <= a:
                    break
            return (b, re_move) 

    def _find_max(self, possible_placements, curr_qvalues): 
        """
        Find the max possible moves in curr_qvalues.
        Return max_values.
        """
        current_max = -np.inf
        row, col = 0, 0
        max_values = []
        for possible_row, possible_col in possible_placements:
            if curr_qvalues[possible_row][possible_col] > current_max:
                current_max = curr_qvalues[possible_row][possible_col]
                max_values = [(possible_row, possible_col)]
            elif curr_qvalues[possible_row][possible_col] == current_max:
                max_values.append((possible_row, possible_col))
        # print(max_values)
        return max_values

    def record(self, curr_state, row, col): 
        """
        Record current state and move place.
        """
        self.history_status.append((curr_state, (row, col)))
        return

    def update(self, result): 
        """
        Update qvalues by using self.history_status.
        """
        if result == 1:
            reward = DRAW_REWARD
        elif result == 2:
            reward = WIN_REWARD
        else:
            reward = LOSS_REWARD
        self.history_status.reverse()
        max_q_value = -np.inf
        for hist in self.history_status: # from the last move to the first move.
            state, move = hist
            q = self.findq(state)
            if max_q_value == -np.inf: # Use reward as qvalues for the last move.
                q[move[0]][move[1]] = reward
            else: # OW, use formular
                q[move[0]][move[1]] = q[move[0]][move[1]] * (1 - self.alpha) + self.alpha * self.gamma * max_q_value
            max_q_value = np.max(q) # Find the max.
        self.history_status = []
        return

    def get_input(self, go, curr_state):
        """
        Get one input find the best move.
        Return move.

        :param go: Go instance.
        """        
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, self.side, test_check = True):
                    possible_placements.append((i,j))

        if not possible_placements:
            return "PASS"
        else:
            row, col = self._select_best_move(curr_state, possible_placements)
            self.record(curr_state, row, col)
            return (row, col)


if __name__ == "__main__":
    if os.path.exists("winner.txt"):
        """
        After game ended, read winner.txt to get winner and side.
        Update qvalues.
        (Need to edit build.sh to let this part work)
        """
        f = open("winner.txt")
        line = f.read(2)
        f.close()
        win = int(line[0])
        side = int(line[1])
        if win == 0:
            result = 1
        elif win == side:
            result = 2
        else:
            result = 0
        player = QLearner(side = side)
        player.read()
        player.update(result)
        player.save(1)
    else:
        """
        Get current state and piece_type.
        Move (action)
        """
        N = 5
        piece_type, previous_board, board = readInput(N)
        go = GO(N)
        go.set_board(piece_type, previous_board, board)
        curr_state = ''.join([str(board[i][j]) for i in range(N) for j in range(N)])
        player = QLearner(side = piece_type)
        player.read()
        action = player.get_input(go, curr_state)
        writeOutput(action)
        player.save(0)

