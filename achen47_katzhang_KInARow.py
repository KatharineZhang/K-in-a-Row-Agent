'''
achen47_katzhang_KInARow.py
Authors: Chen, Ailsa; Zhang, Katharine
  Example:  
    Authors: Smith, Jane; Lee, Laura

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 473, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''

import copy
from agent_base import KAgent
from game_types import State, Game_Type
from winTesterForK import winTesterForK
import google.generativeai as genai
import os
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv() 

genai.configure(api_key=os.getenv("API_KEY"))

AUTHORS = 'Ailsa Chen and Katharine Zhang' 

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'Reginald Gorge'
        if twin: self.nickname += ' Opponent'
        self.long_name = 'The Queen of K-In-A-Row'
        if twin: self.long_name += ' II'
        self.persona = 'Sassy teenager, like Regina George, who uses a lot of slang and is rude'
        if twin:
            self.persona = 'Sassy, mature woman, like Elle Woods, with more professional and mature language'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "X" # e.g., "X" or "O". X - ID = 0, O - ID = 1

    def introduce(self):
        intro = '\nHi, Im Regina George, and Im here to totally *crush* you at Tic Tac Toe.\n'+\
        'Omg your agent is sooo fetch! But not as good as mine.\n'+\
        'I mean, Im basically the queen of everything, so, yeah. Youre going down loser.\n'+\
        'But dont worry, Ill let you have a few moves. You seem like you will need it.\n'+\
        'Anyways, Im kind of busy so Ill give you some advice:\n'+\
        'Im going to win.\n'+\
        'Anyways, toodaloo honey!'
        if self.twin: intro += "By the way, I'm the TWIN.\n"
        return intro

    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(
        self,
        game_type,
        what_side_to_play,
        opponent_nickname,
        expected_time_per_move = 0.1, # Time limits can be
                                      # changed mid-game by the game master.
        utterances_matter=True):      # If False, just return 'OK' for each utterance.

       # Write code to save the relevant information in variables
       
       # local to this instance of the agent.
       # Game-type info can be in global variables.

       self.who_i_play = what_side_to_play
       self.opponent_nickname = opponent_nickname
       self.time_limit = expected_time_per_move
       global GAME_TYPE
       GAME_TYPE = game_type
       print("Currently playing game type: ", game_type.long_name)
       self.my_past_utterances = []
       self.opponent_past_utterances = []
       self.repeat_count = 0
       self.utt_count = 0
       if self.twin: self.utt_count = 5 # Offset the twin's utterances.
       return "Ready!"
   
    # The core of your agent's ability should be implemented here:             
    def makeMove(self, currentState, currentRemark, timeLimit=10000):

        # Here's a placeholder:
        a_default_move = [0, 0] # This might be legal ONCE in a game,
        # if the square is not forbidden or already occupied.
        newState = copy.deepcopy(currentState)      

        alpha = float('-inf')
        beta = float('inf')

        maxV = float('-inf')

        depth = GAME_TYPE.k
        # depth = int(GAME_TYPE.k / 2)
        maxAction = [0,0]   
        
        s_a_pair = successors_and_moves(newState)
        maxSucc = s_a_pair[0][0]
        for i in range(0, len(s_a_pair[0])):
            successor = s_a_pair[0][i]
            # print("Max state:", successor)
            action = s_a_pair[1][i]
            # print("action", action)
            currV = self.minimax(successor, depth, alpha, beta, 1 )
            # print("currV", currV)
            if currV > maxV:
                maxV = currV
                maxAction = action
                maxSucc = successor
                alpha = max(alpha, currV) # update alpha
        model = genai.GenerativeModel("gemini-1.5-flash")
        if not self.twin:
            if currV == 0:
                response = model.generate_content("Write a sassy, angry, one-sentence response as Regina George losing a tic tac toe game.").text
            elif currV < 10:
                response = model.generate_content("Write a sassy, taunting, one sentence remark as Regina George winning tic tac toe").text
            else:
                response = model.generate_content("Write a sassy remark about the tic tac toe game currently, as Regina Georege, in one sentence only").text
        else:
            if currV == 0:
                response = model.generate_content("Write a shy, disappointed, one-sentence response as a fluttershy losing a tic tac toe game.").text
            elif currV < 100:
                response = model.generate_content("Write a shy, happy, one sentence remark as fluttershy winning tic tac toe").text
            else:
                response = model.generate_content("Write a shy, extremely happy, one-sentence remark about a tic tac toe game").text
        model = genai.GenerativeModel("gemini-1.5-flash")
        if not self.twin:
            if currV == 0:
                response = model.generate_content("Write a sassy, angry, one-sentence response as Regina George losing a tic tac toe game.").text
            elif currV < 10:
                response = model.generate_content("Write a sassy, taunting, one sentence remark as Regina George winning tic tac toe").text
            else:
                response = model.generate_content("Write a sassy remark about the tic tac toe game currently, as Regina Georege, in one sentence only").text
        else:
            if currV == 0:
                response = model.generate_content("Write a shy, disappointed, one-sentence response as a fluttershy losing a tic tac toe game.").text
            elif currV < 100:
                response = model.generate_content("Write a shy, happy, one sentence remark as fluttershy winning tic tac toe").text
            else:
                response = model.generate_content("Write a shy, extremely happy, one-sentence remark about a tic tac toe game").text
            
        print(response)
        
        return [[maxAction, maxSucc], response]
    

    # The main adversarial search function:
    def minimax(self,
            state,
            depthRemaining,
            alpha=None,
            beta=None,
            agentID=None):
        #pruning=False, zHashing=None
        # print("Current state \n", state)
        if depthRemaining == 0:
            # print("in depthRemaining,", type(state))
            value = self.staticEval(state)
            #print("after staticEval, ", value)
            # print("after staticEval,", type(state))
            return value
        if agentID == 0: 
            return self.maxValue(state, depthRemaining, alpha, beta, 0)
        if agentID == 1:
            return self.minValue(state, depthRemaining, alpha, beta, 1)
        return
        

    def maxValue(self, state, depth, alpha, beta, agentID):
        # print("MAX IS CALLED \n")
        # print("AGENTID", agentID)
        v = float('-inf')
        s_a_pair = successors_and_moves(state)
        for i in range(0, len(s_a_pair[0])):
            successor = s_a_pair[0][i]
            # print("Max state:", successor)
            action = s_a_pair[1][i]
            # print("inside maxVal:", type(successor))
            currV = self.minimax(successor, depth - 1, alpha, beta, 1) # ghost plays next!
            # print("after self eval in maxVal,", type(successor))
            v = max(v, currV) # only update value if it's the max
            if v >= beta:  # if value is greater than beta, we want to prune
                # print("entered if statement")
                return v
            alpha = max(alpha, v) # update alpha if it's value is > than it
        return v 

    def minValue(self, state, depth, alpha, beta, agentID):
        # print("MIN IS CALLED \n")
        # print("AGENTID", agentID)
        v = float('inf')

        s_a_pair = successors_and_moves(state)
        for i in range(0, len(s_a_pair[0])):
            successor = s_a_pair[0][i]
            # print("Max state:", successor)
            action = s_a_pair[1][i]
            # print("inside minValue:", type(successor))
            currV = self.minimax(successor, depth - 1, alpha, beta, 0)
            #print("after self eval in maxVal,", type(successor))
            #print("currV inside minVal:", currV)
            v = min(v, currV)
            if v <= alpha:
                # print("entered if statement")
                return v
            beta = min(beta, v)
        return v
    

    def staticEval(self, state):
        score = 0
        board = state.board
        n = len(board)        # Number of rows
        m = len(board[0])     # Number of columns
        k = GAME_TYPE.k          # Number of marks in a row to win
        
        # Evaluate all rows, columns, and diagonals
        for i in range(n):
            # Check row
            score += self.evaluateLine(state.board[i])
        
        for j in range(m):
            # Check column
            col = [state.board[i][j] for i in range(n)]
            score += self.evaluateLine(col)
        
        # Check diagonals (from top-left to bottom-right)
        for start in range(n):
            diag1 = []
            diag2 = []
            for i in range(min(n-start, m)):
                diag1.append(state.board[start + i][i])
                diag2.append(state.board[start + i][m - 1 - i])
            score += self.evaluateLine(diag1)
            score += self.evaluateLine(diag2)
        
        # Check diagonals (from top-right to bottom-left)
        for start in range(n):
            diag1 = []
            diag2 = []
            for i in range(min(n-start, m)):
                diag1.append(state.board[start + i][m - 1 - i])
                diag2.append(state.board[start + i][i])
            score += self.evaluateLine(diag1)
            score += self.evaluateLine(diag2)
        
        return score

    def evaluateLine(self, line):
        """
        Helper function to evaluate a single line (row, column, or diagonal).
        It will return a score based on the number of consecutive 'X' and 'O' marks.
        """
        k = GAME_TYPE.k 

        x_count = 0  # Consecutive 'X'
        o_count = 0  # Consecutive 'O'
        empty_count = 0  # Number of empty spaces
        x_empty = 0  # Number of empty spaces for 'X' to complete a line
        o_empty = 0  # Number of empty spaces for 'O' to complete a line
        
        # Count 'X', 'O' and empty spaces in the line
        for cell in line:
            if cell == 'X':
                x_count += 1
            elif cell == 'O':
                o_count += 1
            elif cell == ' ':
                empty_count += 1
                x_empty += 5  # Potential for 'X' to complete a line
                o_empty += 5  # Potential for 'O' to complete a line
            
        
        score = 0
        
        # Win-by-one-move heuristic for 'X' (agent can win in one move)
        if x_count == k - 1 and x_empty == 1:
            score += 100  # Reward 'X' for winning in the next move
        
        # Win-by-one-move heuristic for 'O' (we need to block 'O' from winning in one move)
        if o_count == k - 1 and o_empty == 1:
            score -= 100 # Penalize for opponent's near win
        
        # If line has k consecutive 'X', it's a win for 'X'
        if x_count == k:
            return 1000  # High value for winning line of 'X'
        # If line has k consecutive 'O', it's a win for 'O'
        elif o_count == k:
            return -1000  # High negative value for winning line of 'O'
        
        # Evaluate line based on possible future plays
        if x_count > 0:
            score += x_count * 10  # Positive score for consecutive 'X'
        if o_count > 0:
            score -= o_count * 10  # Negative score for consecutive 'O'
        
        # Factor in the empty spaces (potential for completing the line)
        if empty_count > 0:
            score += empty_count * 2  # Encourage completing the line
        
        return score


    #?Additional factors: We want to check how many X's are in a row and weight differently
    #? If k-1 X's in a row, have a greater penalty than if we have like k-4 in a row or something
    #? Can model this after WE2, with 100,10 and 1 weights
    
    # def staticEval(self, state):
    #     # print("in static eval")
    #     board = state.board
    #     n = len(board)        # Number of rows
    #     m = len(board[0])     # Number of columns
    #     k = GAME_TYPE.k          # Number of marks in a row to win

    #     x_score = 0
    #     o_score = 0
    #     open_ends_x = 0
    #     open_ends_o = 0
    #     x_one_move_away = 0
    #     o_one_move_away = 0
    #     x_pieces = 0
    #     o_pieces = 0
    #     center_control = 0
    #     block_threat_score = 0
        

    #     # Define directions for rows, columns, and diagonals
    #     directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

    #     # Helper function to check if a position is within bounds
    #     # r - row
    #     # c - col
    #     def in_bounds(r, c):
    #         return 0 <= r < n and 0 <= c < m

    #     # Helper function to check if a line (row, column, diagonal) can form a win
    #     line_cache = {}

    #     def get_line(r, c, dr, dc, player):
    #         # Check cache first
    #         if (r, c, dr, dc, player) in line_cache:
    #             return line_cache[(r, c, dr, dc, player)]
            
    #         # Otherwise compute and store the result
    #         count = 0
    #         open_ends = 0
    #         for i in range(k):
    #             newR = r + i * dr
    #             newC = c + i * dc
    #             if in_bounds(newR, newC):
    #                 if board[newR][newC] == player:
    #                     count += 1
    #                 elif board[newR][newC] == ' ':
    #                     open_ends += 1
    #             else:
    #                 line_cache[(r, c, dr, dc, player)] = (0, 0)  # Out of bounds
    #                 return (0, 0)
    #         line_cache[(r, c, dr, dc, player)] = (count, open_ends)
    #         return (count, open_ends)

    #     # Evaluate every potential line for both players
    #     for r in range(n):
    #         for c in range(m):
    #             if board[r][c] == 'X':
    #                 x_pieces += 1
    #             elif board[r][c] == 'O':
    #                 o_pieces += 1

    #             # # **Win-In-One Move Threat Heuristic**: Check if 'X' or 'O' can win in the next move
    #             # for dr, dc in directions:
    #             #     count_x, open_ends_x_line = get_line(r, c, dr, dc, 'X')
    #             #     count_o, open_ends_o_line = get_line(r, c, dr, dc, 'O')

    #             #     # Use the values directly in your heuristics
    #             #     if count_x == k - 1 and open_ends_x_line == 1:
    #             #         x_one_move_away += 1000
    #             #     if count_o == k - 1 and open_ends_o_line == 1:
    #             #         o_one_move_away += 1000

    #             # Check all possible directions for lines of length k
    #             for dr, dc in directions:
    #                 count_x, open_ends_x_line = get_line(r, c, dr, dc, 'X')
    #                 count_o, open_ends_o_line = get_line(r, c, dr, dc, 'O')

    #                 # Use the values directly in your heuristics


    #                 # Heuristics for 'X'
    #                 if count_x > 0:  # 'X' has a potential win
    #                     x_score += (count_x * count_x)
    #                 if open_ends_x_line > 0:
    #                     open_ends_x += open_ends_x_line * 5

    #                 # Heuristics for 'O'
    #                 if count_o > 0:  # 'O' has a potential win
    #                     o_score -= (count_o * count_o)
    #                 if open_ends_o_line > 0:
    #                     open_ends_o += open_ends_o_line * 5

    #                 # **Block and Threat Heuristic**: Block opponent's potential win
    #                 if count_o == k - 1 and open_ends_o_line == 1:  # 'O' is one move away from winning
    #                     block_threat_score -= 100  # Reward block for 'X' player
    #                 if count_x == k - 1 and open_ends_x_line == 1:  # 'X' is one move away from winning
    #                     block_threat_score += 100  # Reward 'X' player for threatening to win

    #             # **Center and Edge Control Heuristic**: Increase score for controlling the center
    #             if (r, c) == (n//2, m//2):
    #                 center_control += 100 if board[r][c] == 'X' else -100
    #             # Increase score for occupying edges or corners
    #             elif r in [0, n-1] or c in [0, m-1]:
    #                 if board[r][c] == 'X':
    #                     x_score += 20
    #                 elif board[r][c] == 'O':
    #                     o_score += 20

                

    #     # Evaluate based on piece density
    #     piece_density = x_pieces - o_pieces

    #     # Closing board evaluation (whether the board is near being filled up)
    #     filled_cells = sum(1 for row in board for cell in row if cell != ' ')
    #     remaining_cells = n * m - filled_cells

    #     if remaining_cells == 1:
    #         if piece_density > 0:
    #             x_score += 50
    #         else:
    #             o_score -= 50

    #     # Final evaluation combining all factors
    #     eval_score = (
    #         x_score - o_score
    #         + open_ends_x - open_ends_o
    #         + piece_density * 2
    #         + x_one_move_away - o_one_move_away
    #         + center_control
    #         + block_threat_score * 2
    #     )

    #     # print("end")
    #     return eval_score

        
    

# determines which player it is - X or O
def other(p):
    if p=='X': return 'O'
    return 'X'
# simulates a single move
def do_move(state, i, j, o):
    news = State(old=state)
    news.board[i][j] = state.whose_move
    news.whose_move = o
    return news
# Gets all successors and their associated moves/actions given the current state
def successors_and_moves(state):
    # print("in successor and moves: ", type(state))
    b = state.board
    p = state.whose_move
    o = other(p)
    new_states = []
    moves = []
    mCols = len(b[0])
    nRows = len(b)

    for i in range(nRows):
        for j in range(mCols):
            if b[i][j] != ' ': continue
            news = do_move(state, i, j, o)
            # print("New state:\n", news)
            new_states.append(news)
            moves.append([i, j])
    # print("at end of s/m:", type(new_states[0]))
    return [new_states, moves]

   
