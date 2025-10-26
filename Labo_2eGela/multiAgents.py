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


import random

import util
from game import Agent
from util import manhattanDistance


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
        foodList = newFood.asList()
        closestFood = min([manhattanDistance(newPos, food) for food in foodList]) if foodList else 0
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        closestGhost = min(ghostDistances) if ghostDistances else float('inf')
        if (closestGhost > 10):
            closestGhost = 10
        elif closestGhost <= 1:
            return -float('inf')  # Avoid states where Pacman is too close to a ghost

        return successorGameState.getScore() + (1 / (closestFood + 1)) * 10 - (closestGhost) * 2
        #return successorGameState.getScore() +  (1 / (closestFood + 1)) * 10 - (closestGhost) * 2 #Tendr�is que comentar esta linea y devolver el valor que calculeis


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
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, game_state):
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
        num_agents = game_state.getNumAgents()
        minimax_depth = self.depth * num_agents

        def minimax(agent_index, depth, state):
            if depth == minimax_depth or state.isWin() or state.isLose(): # evalua la situacion
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent_index)
            if not legal_actions:
                return self.evaluationFunction(state)

            if agent_index == 0:  # maximizador (Pacman)
                value = -float('inf')
                for action in legal_actions:
                    successor = state.generateSuccessor(agent_index, action)
                    value = max(value, minimax(1, depth + 1, successor))
                return value
            else:  # minimizador (fantasmas)
                next_agent = (agent_index + 1) % num_agents
                value = float('inf')
                for action in legal_actions:
                    successor = state.generateSuccessor(agent_index, action)
                    value = min(value, minimax(next_agent, depth + 1, successor))
                return value

        # comparador y seleccionador de la mejor accion
        best_score = -float('inf')
        best_action = None
        for action in game_state.getLegalActions(0):
            successor = game_state.generateSuccessor(0, action)
            score = minimax(1, 1, successor)
            if score > best_score or best_action is None:
                best_score = score
                best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Alpha-beta search over plies. We avoid pruning on equality to
        # match the autograder's explored-node set (use strict < and >).
        num_agents = game_state.getNumAgents()
        max_depth = self.depth * num_agents

        def ab_search(agent_index, depth, state, alpha, beta):
            # terminal o profundidad alcanzada
            if depth == max_depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent_index)
            if not legal_actions:
                return self.evaluationFunction(state)

            if agent_index == 0:  # minimizador (Pacman)
                value = -float('inf')
                for action in legal_actions:
                    succ = state.generateSuccessor(agent_index, action)
                    value = max(value, ab_search(1, depth + 1, succ, alpha, beta))
                    # actualiza alpha no usando igualdad
                    if value > alpha:
                        alpha = value
                    if alpha > beta:
                        return value
                return value
            else:  # minimizador (fantasmas)
                next_agent = (agent_index + 1) % num_agents
                value = float('inf')
                for action in legal_actions:
                    succ = state.generateSuccessor(agent_index, action)
                    value = min(value, ab_search(next_agent, depth + 1, succ, alpha, beta))
                    # actualiza beta no usando igualdad
                    if value < beta:
                        beta = value
                    if alpha > beta:
                        return value
                return value

        # comparador y seleccionador de la mejor accion
        best_action = None
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for action in game_state.getLegalActions(0):
            succ = game_state.generateSuccessor(0, action)
            score = ab_search(1, 1, succ, alpha, beta)
            if score > best_score or best_action is None:
                best_score = score
                best_action = action
            # actualiza alpha no usando igualdad
            if score > alpha:
                alpha = score

        return best_action



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
        num_agents = gameState.getNumAgents()
        max_depth = self.depth * num_agents

        def expectimax(agent_index, depth, state):
            if depth == max_depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent_index)
            if not legal_actions:
                return self.evaluationFunction(state)

            if agent_index == 0:
                value = -float('inf')
                for action in legal_actions:
                    succ = state.generateSuccessor(agent_index, action)
                    value = max(value, expectimax(1, depth + 1, succ))
                return value
            else:
                total = 0.0
                next_agent = (agent_index + 1) % num_agents
                for action in legal_actions:
                    succ = state.generateSuccessor(agent_index, action)
                    total += expectimax(next_agent, depth + 1, succ)
                return total / len(legal_actions)

        best_action = None
        best_score = -float('inf')
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            score = expectimax(1, 1, succ)
            if score > best_score or best_action is None:
                best_score = score
                best_action = action

        return best_action



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Esta función de evaluación tiene en cuenta la posición de Pacman, la comida, los fantasmas
    y las cápsulas para calcular una puntuacion que determine el estado del juego.
    1- Calcula la distancia a la comida más cercana
    2- Calcula la distancia al fantasma más cercano y si está asustado o no
    3- Aumenta la puntuación en base a la cantidad de comida restante y la distancia
    4- Disminuye la puntuación si un fantasma está demasiado cerca y no está asustado
    5- Las cápsulas también aumentan la puntuación en función de su distancia

    """
    pacman_pos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    food_list = newFood.asList()

    score = currentGameState.getScore()

    # comida más cercana
    if food_list:
        closest_food = min(util.manhattanDistance(pacman_pos, f) for f in food_list)
        score += 5.0 / (closest_food + 1)
    score -= 2.0 * len(food_list)

    # capsulas
    if currentGameState.getCapsules():
        score += 8.0 / (min(util.manhattanDistance(pacman_pos, c) for c in currentGameState.getCapsules()) + 1)

    # manejo del fantasma
    for ghost, tiempo in zip(newGhostStates, newScaredTimes):
        distancia = util.manhattanDistance(pacman_pos, ghost.getPosition())
        if tiempo == 0:
            if distancia <= 1:
                return -999999
            score -= 4.0 / distancia
        else:
            score += 2.0 / (distancia + 1) + 0.5 * tiempo

    return score

# Abbreviation
better = betterEvaluationFunction
