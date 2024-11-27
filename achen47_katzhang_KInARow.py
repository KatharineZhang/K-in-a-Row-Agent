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
        self.nickname = 'Reginald George'
        if twin: self.nickname += ' the Twin'
        self.long_name = 'The Queen of K-In-A-Row'
        if twin: self.long_name += ' II'
        self.persona = 'Sassy teenager, like Regina George, who uses a lot of 2000s slang, is passive aggressive, and rude'
        if twin:
            self.persona = 'Sassy, mature woman, like Elle Woods, with more professional and mature language'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "X" # e.g., "X" or "O". X - ID = 0, O - ID = 1

    def introduce(self):
        intro = '\nHi, Im Reginald George, and Im here to totally *crush* you at K-in-a-row!.\n'+\
        'But dont worry, Ill let you have a few moves. You seem like you will need it.\n'
        if self.twin: 
            intro = '\nHi, Im Reginald George \'s Twin, and I play to win darling. You\'re no match for my wits!\n'
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

        # depth = GAME_TYPE.k
        depth = int(GAME_TYPE.k / 2)
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
        
        #fix the conditionals for this
        if maxV <= 0:
            response = model.generate_content("Write a 2-sentence max remark to being in a losing position for a  k-in-a-row game with this tone:" + self.persona).text
        elif maxV >= 100000:
            response = model.generate_content("Write a 2-sentence max remark to having a major advantage in a k-in-a-row game with this tone:" + self.persona).text
        else:
            response = model.generate_content("Write a 2-sentence max remark to the other player a  k-in-a-row game with this tone:" + self.persona).text
        
        
        # response = "test"
        
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
            value = self.staticEval(state)
            # print("after staticEval, ", value)
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
        max_x_count = 0
        max_o_count = 0
        
        # Count 'X', 'O' and empty spaces in the line
        for i in range(len(line)):
            cell = line[i]
            if cell == 'X':
                x_count += 1
                o_count = 0
                max_x_count = max(max_x_count,x_count)
                
            #this is supposed to check like if the blank spaces are next to X bc x_ should be worth more than like xo_
            #it also represents a potential completion of the line
            
            # if cell == 'X' and i+1 < len(line) and line[i+1] == ' ':
            #     x_empty += 1
            #     x_count = 0
            #     o_count = 0
            if cell == 'O':
                o_count += 1
                x_count = 0
                max_o_count = max(max_o_count,o_count)
                
            # if cell == 'O' and i+1 < len(line) and line[i+1] == ' ':
            #     o_empty += 1
            #     o_count = 0
            #     x_count = 0
            if cell == ' ':
                o_count = 0
                x_count = 0
            
        x_count= max_x_count
        o_count = max_o_count
        score = 0
        
        # Win-by-one-move heuristic for 'X' (agent can win in one move)
        if x_count + x_empty == k:
            score += x_empty * 10
        if o_count + o_empty == k:
            score -= o_empty * 10
        
        if x_count == k - 1:
            return 100000 # Reward 'X' for winning in the next move
        
        # Win-by-one-move heuristic for 'O' (we need to block 'O' from winning in one move)
        if o_count == k - 1:
            return -100000 # Penalize for opponent's near win
        
        # If line has k consecutive 'X', it's a win for 'X'
        if x_count == k:
            return 10000000000000  # High value for winning line of 'X'
        # If line has k consecutive 'O', it's a win for 'O'
        elif o_count == k:
            return -10000000000000  # High negative value for winning line of 'O'
        
        # Evaluate line based on possible future plays
        if x_count > 0:
            score += x_count * 10  # Positive score for consecutive 'X'
            # score += x_empty *2
        if o_count > 0:
            score -= o_count * 10  # Negative score for consecutive 'O'
            # score += o_empty *2

        
        return score
    

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

   
