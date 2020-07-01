# 2516821162：Yijing Yang
import pickle
import numpy as np
import os
import random
import sys
from collections import Counter
from copy import deepcopy
from queue import Queue

WIN_REWARD = 1.0
DRAW_REWARD = 0.5
LOSS_REWARD = -0.2

def readInput(n, path="input.txt"):
    """
    Read input file.
    """

    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return piece_type, previous_board, board

def writeOutput(result, path="output.txt"):
    """
    Write output file.
    """
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)

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

    def record(self, curr_state, row, col): 
        """
        Record current state and move place.
        """
        self.history_status.append((curr_state, (row, col)))
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

    def findq_noadd(self, state): 
        """
        Find curr_qvalues for current state.
        Return curr_qvalues
        """
        if self.side == 1:
            side_qvalues = self.X_qvalues
        else:
            side_qvalues = self.O_qvalues
        if state not in side_qvalues:
            initial_q = np.zeros((5, 5))
            initial_q.fill(self.initial_value)
            return initial_q
        return side_qvalues[state]

    def _select_best_move(self, curr_state, possible_placements, go): 
        """
        Find the max possible move in curr_qvalues.
        If max is not unique, use minmax to find the best move.
        OW, max is the best move.
        Return best move.
        """
        curr_qvalues = self.findq(curr_state)
        max_values = self._find_max(go, possible_placements, curr_qvalues)
        if len(max_values) == 1:
            re_move = max_values[0]
        elif max_values == []:
            return "PASS"
        else:
            re_move = self.alphabeta(go, 3, self.side, max_values)
        row = re_move[0]
        col = re_move[1]
        return (row, col)

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
        #self.history_status = []
        return

    def alphabeta(self, go, depth, side, max_values):
        """
        Test all max_values by ab-prunning to see who gets the best score.
        Return best move.
        """
        a = -np.inf
        b = np.inf
        best = a
        re_move = max_values[0]

        for next_move in max_values: 
            go.make_move(next_move, side)
            v = self.alphabeta_min(go, depth, a, b, 3 - side, next_move)
            go.cancel_move()
            if v > best:
                best = v
                re_move = next_move
            a = max(a, best)
            if a >= b: # Prunning
                break
        return re_move

    def alphabeta_min(self, go, depth, a, b, side, move):

        # Reach the max depth.
        if depth == 0:
            return self.evaluate(go, side)

        # Game end.         
        if go.game_end(move):
            return self.evaluate(go, side)    

        possible_placements = self.find_possible_placements(go, side)
        if possible_placements[0] == "PASS" :
            max_moves = possible_placements
        else:
            state = go.board
            str_state = ''.join([str(state[i][j]) for i in range(N) for j in range(N)])
            curr_qvalues = self.findq_noadd(str_state)
            max_moves = self._find_max(go, possible_placements, curr_qvalues) 

        best = np.inf
        for next_move in max_moves:
            go.make_move(next_move, side)
            v = self.alphabeta_max(go, depth - 1, a, b, 3 - side, next_move)
            go.cancel_move()
            best = min(v, best)
            b = min(b, best)
            if a >= b: # Prunning
                break
        return best

    def alphabeta_max(self, go, depth, a, b, side, move):

        # Reach the max depth.
        if depth == 0:
            return self.evaluate(go, side)

        # Game end.         
        if go.game_end(move):
            return self.evaluate(go, side) 

        possible_placements = self.find_possible_placements(go, side)
        if possible_placements[0] == "PASS" :
            max_moves = possible_placements
        else:
            state = go.board
            str_state = ''.join([str(state[i][j]) for i in range(N) for j in range(N)])
            curr_qvalues = self.findq_noadd(str_state)
            max_moves = self._find_max(go, possible_placements, curr_qvalues) 

        best = -np.inf
        for next_move in max_moves:
            go.make_move(next_move, side)
            v = self.alphabeta_min(go, depth - 1, a, b, 3 - side, next_move)
            go.cancel_move()
            best = max(v, best)
            a = max(a, best)
            if a >= b: # Prunning
                break
        return best

    def evaluate(self, go, side):
        score = Counter()
        c = Counter()

        score[1] = 0
        score[2] = 3 * 15

        for i in range(go.size):
            for j in range(go.size):

                ptype = go.board[i][j]
                if ptype != 0:
                    score[ptype] += 15
                    c[ptype] += 1

                    liberty = 0
                    my = 0
                    not_my = 0
                    neighbors = go.detect_neighbor(i, j)
                    space = 0
                    for p, q in neighbors:
                        if go.board[p][q] == 0:
                            liberty += 1
                            empty_neighbors = go.detect_neighbor(p, q)
                            empty_liberty = 0
                            for x, y in empty_neighbors:
                                if go.board[x][y] == 0:
                                    empty_liberty += 1
                            if  empty_liberty > 0:
                                space = 1
                        elif go.board[p][q] == ptype:
                            my += 1
                            next_neighbors = go.detect_neighbor(p, q)
                            next_liberty = 0
                            next_my = 0
                            for x, y in next_neighbors:
                                if go.board[x][y] == 0:
                                    next_liberty += 1
                                if go.board[x][y] != ptype:
                                    next_my += 1
                            if next_liberty > 0:
                                space = 1
                            if next_my == len(next_neighbors) - 1:
                                score[ptype] += 3
                        else:
                            not_my += 1
                            notmy_liberty = 0
                            zhanling = 0
                            notmy_neighbors = go.detect_neighbor(p, q)
                            for x, y in notmy_neighbors:
                                if go.board[x][y] == 0:
                                    notmy_liberty += 1
                                elif go.board[x][y] == ptype:
                                    zhanling += 1
                            if  notmy_liberty <= 3:
                                score[ptype] += 1
                            elif notmy_liberty == 1:
                                score[ptype] += 1
                            elif notmy_liberty == 0:
                                score[ptype] += 1
                            if zhanling == len(notmy_neighbors) - 1:
                                score[ptype] += 2

                    if space == 0:
                        score[ptype] -= 5
                        score[3 - ptype] += 5

                    if len(neighbors)- my <= 2:
                        score[ptype] -= 1



                    if liberty == 1:
                        if not_my >= 2:
                            score[ptype] -=0.5
                    elif liberty == 2:
                        if not_my < 2:
                            score[ptype] += 1.5
                        else:
                            score[ptype] += 1

                    elif liberty == 3:
                        score[ptype] += 2

                    else:
                        score[ptype] -=0.5
                    
                else:
                    neighbors = go.detect_neighbor(i, j)
                    counter = Counter()
                    for p, q in neighbors:
                        counter[go.board[p][q]] += 1
                    if counter[1] == len(neighbors):
                        score[1] += 3
                    elif counter[2] == len(neighbors):
                        score[2] += 3

        if c[1] == 12:
            score[1] += 50
        if c[2] >= 9:
            score[2] += 50

        elif c[1] <= 11 & c[2] >= 8:
            score[2] += 50

        return score[self.side] - score[3 - self.side]

    def _find_max(self,go, possible_placements, curr_qvalues): 
        """
        Find the max possible moves in curr_qvalues.
        Return max_values.
        
        """
        current_max = -np.inf
        row, col = 0, 0
        max_values = []
        count = len(possible_placements)
        for possible_row, possible_col in possible_placements:
            if curr_qvalues[possible_row][possible_col] > current_max:
                current_max = curr_qvalues[possible_row][possible_col]
                max_values = [(possible_row, possible_col)]
            elif curr_qvalues[possible_row][possible_col] == current_max:
                neibghors = go.detect_neighbor(possible_row, possible_col, 1)
                count -= 1
                if sum([go.board[p][q] for p, q in neibghors]) != 0 or count < 15:
                    max_values.append((possible_row, possible_col))
                    count += 1
        return max_values
        

    def get_input(self, go, curr_state):
        """
        Get one input find the best move.
        Return move.
        """   
        possible_placements = self.find_possible_placements(go, self.side)
        if possible_placements[0] == 'PASS':
            return possible_placements[0]
        elif len(possible_placements) == 25:
            return (2, 2)      

        else:
            move = self._select_best_move(curr_state, possible_placements, go)
            if move == "PASS":
                return move
            else:
                row = move[0]
                col = move[1]
                self.record(curr_state, row, col)
                return (row, col)

    def find_possible_placements(self, go, piece_type):
        """
        Get all possible placements and return.
        """
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type):
                    possible_placements.append((i,j))
        if len(possible_placements) == 0:
            return ['PASS']
        random.shuffle(possible_placements)
        return possible_placements



class GO:
    def __init__(self, n):
        self.size = n
        self.board = np.zeros((5, 5))
        self.previous_board = deepcopy(board)
        self.died_pieces = []
        self.n_move = 0
        self.max_move = n * n - 1
        self.history_status = []

    def set_board(self, piece_type, previous_board, board):
        """
        Initialize board status.
        """
        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        self.history_status.append(previous_board)
        self.previous_board = previous_board
        self.board = board
        return

    def detect_neighbor(self, i, j, flag = 0):
        """
        Detect whether 4 or 8 neighbors of a given stone.
        Return neighbors
        """
        neighbors = []
        if flag == 0:
            dx = [0, 0, 1, -1]
            dy = [1, -1, 0, 0]
        else:
            dx = [0, 0, 1, 1, 1, -1, -1, -1]
            dy = [1, -1, 1, 0, -1, 1, 0, -1]
        for k in range(len(dx)):
            if 0 <= i + dx[k] < self.size and 0 <= j + dy[k] < self.size:
                neighbors.append((i + dx[k], j + dy[k]))
        return neighbors

    def compare_board(self, board1, board2):
        """
        Compare two boards.
        Return result.
        """
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        """
        Return copy the current board for test.
        """
        return deepcopy(self)

    def valid_place_check(self, i, j, piece_type):
        """
        Return if i,j is valid for piece_type
        """

        # Check if the place is in the board range
        if not (i >= 0 and i < self.size):
            return False
        if not (j >= 0 and j < self.size):
            return False
        
        # Check if the place already has a piece
        if self.board[i][j] != 0:
            return False

        # Check if the place has liberty
        self.board[i][j] = piece_type
        dead_pieces = set()
        self.check_liberty(i, j, dead_pieces)
        self.board[i][j] = 0
        if len(dead_pieces) == 0:
            return True

        # Copy the board for testing
        test_go = self.copy_board()
        test_go.board[i][j] = piece_type

        # If not, remove the died pieces of opponent and check again
        dead_pieces = test_go.remove_died_pieces(3 - piece_type)
        if not dead_pieces:
            return False
        elif self.died_pieces and self.compare_board(self.previous_board, test_go.board):
            return False
        return True

    def game_end(self, action="MOVE"):
        """
        Return if game ended.
        """
        if self.n_move >= self.max_move:
            return True
        if self.compare_board(self.previous_board, self.board) and action == "PASS":
            return True
        return False

    def check_liberty(self, i, j, died_pieces):
        """
        Check liberty
        """
        has_liberty = False
        queue = Queue()
        queue.put((i, j))
        visited = set()
        while not queue.empty():
            x, y = queue.get()
            visited.add((x, y))
            for p, q in self.detect_neighbor(x, y):
                if self.board[p][q] == 0:
                    has_liberty = True
                elif self.board[x][y] == self.board[p][q] and (p, q) not in visited:
                    queue.put((p, q))

        if not has_liberty:
            died_pieces |= visited

    def find_all_died_pieces(self, piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.
        '''
        died_pieces = set()
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == piece_type and (i, j) not in died_pieces:
                    self.check_liberty(i, j, died_pieces)
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.
        '''
        died_pieces = self.find_all_died_pieces(piece_type)
        for piece in died_pieces:
            self.board[piece[0]][piece[1]] = 0
        return died_pieces

    def make_move(self, action, piece_type):
        """
        Make a move.
        """
        self.n_move += 1
        self.history_status.append(deepcopy(self.board))
        self.previous_board = self.history_status[len(self.history_status) - 1]
        if action != 'PASS':
            self.board[action[0]][action[1]] = piece_type
        self.died_pieces = list(self.remove_died_pieces(3 - piece_type))
        return

    def cancel_move(self):
        """
        Back to the status before the move.
        """
        self.n_move -= 1
        self.board = self.history_status.pop()
        self.previous_board = self.history_status[len(self.history_status) - 1]



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

