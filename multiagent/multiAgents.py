# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        closetFood = float('inf')
        for food in newFood.asList():
            dis = util.manhattanDistance(newPos, food)
            if dis < closetFood:
                closetFood = dis  # Select the closet food to eat
        score = score + 1 / closetFood  # The closer to the food, the higher the score will be
        closetGhost = float('inf')
        for ghost in newGhostStates:
            dis = util.manhattanDistance(newPos, ghost.getPosition())
            if dis < closetGhost:
                if dis <= 1:  # If pacman is too close to the ghost, the score will be very low
                    score = score - 50
                else:
                    closetGhost = dis
        score = score - 1 / closetGhost  # The closer to the ghost, the lower the score will be
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        """
        MAX~M
        MIN~m
                              M                     //=Max(All min-value successor) => Best move
                           /     \                  |
                          m        m                | Each minnode: value =Min(All min-value successor if next agent is ghost)
                         /.\      /.\               | Number of MinLayer = Number of MinAgent
                        m   m    m   m              | Each minnode: value =Min(All max-value successor if next agent is Pacman)
                       /.\ /.\  /.\ /.\             |
                      M  M M M M M M  M             | Depth + 1, return value if reach depth or win/lose state

        """

        def minimax(state):  # Calculate the minimax value to select best move
            minimax = float('-inf')
            move = Directions.STOP
            for nextAction in state.getLegalActions(0):
                value = minvalue(1, 0, state.generateSuccessor(0, nextAction))
                if value > minimax:
                    minimax = value
                    move = nextAction
            return move

        def minvalue(agentID, curDep, state):  # Calculate the min value
            if state.isWin() or state.isLose() or curDep == self.depth:
                return self.evaluationFunction(state)
            if agentID == state.getNumAgents() - 1:  # => next agent is Pacman
                v = float('inf')
                for action in state.getLegalActions(agentID):
                    v = min(v, maxvalue(curDep + 1, state.generateSuccessor(agentID, action)))
            else:  # => next agent is ghost
                v = float('inf')
                for action in state.getLegalActions(agentID):
                    v = min(v, minvalue(agentID + 1, curDep, state.generateSuccessor(agentID, action)))
            return v

        def maxvalue(curDep, state):  # Calculate the max value
            if state.isWin() or state.isLose() or curDep == self.depth:
                return self.evaluationFunction(state)
            v = float('-inf')
            for action in state.getLegalActions(0):
                v = max(v, minvalue(1, curDep, state.generateSuccessor(0, action)))
            return v

        return minimax(gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def abpruning(state):  # Calculate the value at root to select best move
            alpha = float('-inf')
            beta = float('inf')
            v = float('-inf')
            move = Directions.STOP
            for nextAction in state.getLegalActions(0):
                value = minvalue(1, 0, state.generateSuccessor(0, nextAction), alpha, beta)
                if value > v:
                    v = value
                    move = nextAction
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return move

        def minvalue(agentID, curDep, state, alpha, beta):  # Calculate the min value
            if state.isWin() or state.isLose() or curDep == self.depth:
                return self.evaluationFunction(state)
            if agentID == state.getNumAgents() - 1:
                v = float('inf')
                for action in state.getLegalActions(agentID):
                    v = min(v, maxvalue(curDep + 1, state.generateSuccessor(agentID, action), alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v
            else:
                v = float('inf')
                for action in state.getLegalActions(agentID):
                    v = min(v, minvalue(agentID + 1, curDep, state.generateSuccessor(agentID, action), alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        def maxvalue(curDep, state, alpha, beta):  # Calculate the max value
            if state.isWin() or state.isLose() or curDep == self.depth:
                return self.evaluationFunction(state)
            v = float('-inf')
            for action in state.getLegalActions(0):
                v = max(v, minvalue(1, curDep, state.generateSuccessor(0, action), alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        return abpruning(gameState)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        """
        MAX~M
        ExpectedValue~E
                              M                     //=Max(All min-value successor) => Best move
                           /     \                  |
                          E       E                 | Each ExNode: value =Average(All Ex-value successor if next agent is ghost)
                         /.\      /.\               | Number of ExLayer = Number of MinAgent(not optimal)
                        E   E    E   E              | Each ExNode: value =Average(All max-value successor if next agent is Pacman)
                       /.\ /.\  /.\ /.\             | 
                      M  M M M M M M  M             | Depth + 1, return value if reach depth or win/lose

        """

        def expectimax(state):  # Select Best Move from action has max value
            v = float('-inf')
            move = Directions.STOP
            for nextAction in state.getLegalActions(0):
                value = expectvalue(1, 0, state.generateSuccessor(0, nextAction))
                if value > v:
                    v = value
                    move = nextAction
            return move

        def expectvalue(agentID, curDep, state):  # Calculate expected value
            if state.isWin() or state.isLose() or curDep == self.depth:
                return self.evaluationFunction(state)
            if agentID == state.getNumAgents() - 1:
                sum = 0
                numOfAcs = 0
                for action in state.getLegalActions(agentID):
                    sum = sum + maxvalue(curDep + 1, state.generateSuccessor(agentID, action))
                    numOfAcs = numOfAcs + 1
                return sum / numOfAcs
            else:
                sum = 0
                numofAcs = 0
                for action in state.getLegalActions(agentID):
                    sum = sum + expectvalue(agentID + 1, curDep, state.generateSuccessor(agentID, action))
                    numofAcs = numofAcs + 1
                return sum / numofAcs

        def maxvalue(curDep, state):  # Calculate others maxvalue
            if state.isWin() or state.isLose() or curDep == self.depth:
                return self.evaluationFunction(state)
            v = float('-inf')
            for action in state.getLegalActions(0):
                v = max(v, expectvalue(1, curDep, state.generateSuccessor(0, action)))
            return v

        return expectimax(gameState)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    The evaluation function should evaluate states, rather than actions like reflex agent evaluation function did.
                                               |                   |
                                        /Current state/       /Successor/
    """
    "*** YOUR CODE HERE ***"

    currentPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newCapsule = currentGameState.getCapsules()
    newGhostState = currentGameState.getGhostStates()
    score = currentGameState.getScore()

    closetFood = float('inf')
    for food in newFood.asList():
        dis = util.manhattanDistance(currentPos, food)
        if dis < closetFood:
            closetFood = dis
    score = score + 1 / closetFood
    closetGhost = float('inf')
    for ghost in newGhostState:
        dis = util.manhattanDistance(currentPos, ghost.getPosition())
        if dis < closetGhost and dis != 0:
            if dis <= 2:
                score = score - 50
            else:
                closetGhost = dis
    score = score - 1 / closetGhost

    closetCapsule = float('inf')
    for capsule in newCapsule:  # calculate the closet capsule
        dis = util.manhattanDistance(currentPos, capsule)
        if dis < closetCapsule:
            closetCapsule = dis

        if closetCapsule <= closetGhost and closetCapsule <= 1:  # if the closet capcule is too close and the ghost is so far, Pacman will wait for the ghost to come near to eat the capsule and kill the ghost
            score = score + 50
        else:  # => eat foods
            score = score - 50

    return score


# Abbreviation
better = betterEvaluationFunction
