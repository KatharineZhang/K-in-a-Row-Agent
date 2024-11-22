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

from agent_base import KAgent
from game_types import State, Game_Type
from winTesterForK import winTesterForK

AUTHORS = 'Ailsa Chen and Katharine Zhang' 

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'Nic'
        if twin: self.nickname += '2'
        self.long_name = 'Templatus Skeletus'
        if twin: self.long_name += ' II'
        self.persona = 'bland'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "X" # e.g., "X" or "O". X - ID = 0, O - ID = 1

    def introduce(self):
        intro = '\nMy name is Templatus Skeletus.\n'+\
            '"An instructor" made me.\n'+\
            'Somebody please turn me into a real game-playing agent!\n'
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
        utterances_matter=False):      # If False, just return 'OK' for each utterance.

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
       return "OK"
   
    # The core of your agent's ability should be implemented here:             
    def makeMove(self, currentState, currentRemark, timeLimit=10000):
        print("makeMove has been called")

        print("code to compute a good move should go here.")
        # Here's a placeholder:
        a_default_move = [0, 0] # This might be legal ONCE in a game,
        # if the square is not forbidden or already occupied.
    
        newState = currentState # This is not allowed, and even if
        # it were allowed, the newState should be a deep COPY of the old.    
        newRemark = "I need to think of something appropriate.\n" +\
        "Well, I guess I can say that this move is probably illegal."

        alpha = float('-inf')
        beta = float('inf')

        maxV = float('-inf')

        depth = GAME_TYPE.k
        maxAction = [0,0]
        


        for s_a_pair in successors_and_moves(newState):
            successor = s_a_pair[0]
            action = s_a_pair[1]
            
            currV = self.minimax( successor, depth, alpha, beta, 1 ) # the ghost plays next, with the first ghost being index = 1
            if currV > maxV:
                maxV = currV
                maxAction = action
                alpha = max(alpha, currV) # update alpha
        print("Returning from makeMove")
        return [[maxAction, newState], newRemark]
    

    # The main adversarial search function:
    def minimax(self,
            state,
            depthRemaining,
            alpha=None,
            beta=None,
            agentID=None):
        #pruning=False, zHashing=None

        if depthRemaining == 0:
            return self.staticEval(state)
        if agentID == 0: 
            return self.maxValue(state, depthRemaining, alpha, beta, 0)
        if agentID == 1:
            return self.minValue(state, depthRemaining, alpha, beta, 1)
        return
        

    def maxValue(self, state, depth, alpha, beta, agentID):
        v = float('-inf')
        
        for s_a_pair in successors_and_moves(state):
            successor = s_a_pair[0]
            action = s_a_pair[1]
            currV = self.minimax(successor, depth, alpha, beta, 1) # ghost plays next!
            v = max(v, currV) # only update value if it's the max
            if v > beta:  # if value is greater than beta, we want to prune
                return v
            alpha = max(alpha, v) # update alpha if it's value is > than it
        return v 

    def minValue(self, state, depth, alpha, beta, agentID):
        v = float('inf')

        for s_a_pair in successors_and_moves(state):
            successor = s_a_pair[0]
            action = s_a_pair[1]
            currV = self.minimax(successor, depth - 1 , alpha, beta, 0)
            v = min(v, currV)
            if v < alpha:
                return v
            beta = min(beta, v)
        return v
 

    #we look at each line and look at how many x's have k-l in a row (as long as l is
    # less than k),
    #Also look at how many handicapped spots
    #(might not b good) look at number of positions where we can block a k in a row
    def staticEval(self, state):
        # Checking for Rows for X or O victory. 
        board = state.board
        for row in range(0, 3): 
        
            if board[row][0] == board[row][1] and board[row][1] == board[row][2]: 
            
                if board[row][0] == 'X':
                    return 10
                elif board[row][0] == 'O': 
                    return -10
    
        # Checking for Columns for X or O victory. 
        for col in range(0, 3): 
        
            if board[0][col] == board[1][col] and board[1][col] == board[2][col]: 
            
                if board[0][col]=='X':
                    return 10
                elif board[0][col] == 'O': 
                    return -10
    
        # Checking for Diagonals for X or O victory. 
        if board[0][0] == board[1][1] and board[1][1] == board[2][2]: 
        
            if board[0][0] == 'X': 
                return 10
            elif board[0][0] == 'O': 
                return -10
        
        if board[0][2] == board[1][1] and board[1][1] == board[2][0]: 
        
            if board[0][2] == 'X': 
                return 10
            elif board[0][2] == 'O': 
                return -10
        
        # Else if none of them have won then return 0 
        return 0
    
    # getAll States
    # getNextState based on action: how to do this?????? for success, action in successors_and_moves(state):
    # get all actions
    # use game_types change_turn function
    # define your own agentID


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

   
 
# OPTIONAL THINGS TO KEEP TRACK OF:

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances

