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

genai.configure(api_key="AIzaSyB0yKVjkQUJr-vg_jPvLLPA1I9PdwP8cM4")

AUTHORS = 'Ailsa Chen and Katharine Zhang' 

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.


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
        utterances_matter=False):      # If False, just return 'OK' for each utterance.

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
       self.utt_matters = utterances_matter
       if self.twin: self.utt_count = 5 # Offset the twin's utterances.
       return "Ready!"
   
    # The core of your agent's ability should be implemented here:             
    def makeMove(self, currentState, currentRemark, timeLimit=10000):

        start_time = time.time()
        newState = copy.deepcopy(currentState)      
        alpha = float('-inf')
        beta = float('inf')
        maxV = float('-inf')
        depth = 1
        if time.time() - start_time < timeLimit: 
            depth += 1
        maxAction = [0,0]     
        s_a_pair = successors_and_moves(newState)
        maxSucc = s_a_pair[0][0]
        for i in range(0, len(s_a_pair[0])):
            successor = s_a_pair[0][i]
            action = s_a_pair[1][i]
            currV = self.minimax(successor, depth, alpha, beta, 1 )
            if currV > maxV:
                maxV = currV
                maxAction = action
                maxSucc = successor
                alpha = max(alpha, currV)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        staticVal = self.staticEval(currentState)
        print ("static val", staticVal)
        if self.utt_matters:
            if staticVal <=0: #if our agent is not really winning
                response = model.generate_content("Write a 2-sentence max remark to being in a disadvantage for a  k-in-a-row game with this tone:" + self.persona).text + 'boo'
            elif staticVal >= 100000: #if our agent is winning 
                response = model.generate_content("Write a 2-sentence max remark to having a major advantage in a k-in-a-row game with this tone:" + self.persona).text
            else:
                response = model.generate_content("Write a 2-sentence max remark to the other player a k-in-a-row game with this tone:" + self.persona).text
        else:
            response = "no response"
        
        return [[maxAction, maxSucc], response]
    

    # The main adversarial search function:
    def minimax(self,
            state,
            depthRemaining,
            alpha=None,
            beta=None,
            agentID=None):

        if depthRemaining == 0:
            value = self.staticEval(state)
            return value
        if agentID == 0: 
            return self.maxValue(state, depthRemaining, alpha, beta, 0)
        if agentID == 1:
            return self.minValue(state, depthRemaining, alpha, beta, 1)
        return
        

    def maxValue(self, state, depth, alpha, beta, agentID):

        v = float('-inf')
        s_a_pair = successors_and_moves(state)
        for i in range(0, len(s_a_pair[0])):
            successor = s_a_pair[0][i]
            action = s_a_pair[1][i]
            currV = self.minimax(successor, depth - 1, alpha, beta, 1) 
            v = max(v, currV)
            if v >= beta:
                return v
            alpha = max(alpha, v) 
        return v 

    def minValue(self, state, depth, alpha, beta, agentID):
        v = float('inf')

        s_a_pair = successors_and_moves(state)
        for i in range(0, len(s_a_pair[0])):
            successor = s_a_pair[0][i]
            action = s_a_pair[1][i]
            currV = self.minimax(successor, depth - 1, alpha, beta, 0)
            v = min(v, currV)
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    

    def staticEval(self, state):
        score = 0
        board = state.board
        n = len(board)        
        m = len(board[0])     
        k = GAME_TYPE.k        
        
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

        x_count = 0 
        o_count = 0  
        max_x_count = 0
        max_o_count = 0
        
        # Count 'X', 'O' and empty spaces in the line
        for i in range(len(line)):
            cell = line[i]
            if cell == 'X':
                x_count += 1
                o_count = 0
                max_x_count = max(max_x_count,x_count)
                
            elif cell == 'O':
                o_count += 1
                x_count = 0
                max_o_count = max(max_o_count,o_count)
                
            elif cell == ' ':
                o_count = 0
                x_count = 0
            
        x_count= max_x_count
        o_count = max_o_count
        score = 0
        
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
            score += x_count * x_count
        if o_count > 0:
            score -= o_count * o_count        
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
            new_states.append(news)
            moves.append([i, j])
    return [new_states, moves]

   
