# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
import distanceCalculator

class NullGraphics:
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent:
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    # def chooseAction(self, gameState):
    #     "By default, a BustersAgent just stops.  This should be overridden."
    #     return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food

    ''' Print the layout'''
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move

class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)

    def __str__(self):
        a + b
        a.__add__(b)
        return "Node({})".format(self.position)

def children(parent, position, grid):
    offsets = []
    width, length = len(grid.data), len(grid.data[0])

    for child in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        offi = position[0] + child[0], position[1] + child[1]
        xw, yw = offi
        if width >= xw >= 0 and length >= yw >= 0 and grid[xw][yw] != True:
            offsets.append(Node(parent, offi))


    return offsets


def aStar(start, goal, grid):

    openset = set()
    closedset = set()
    current = Node(None, start)
    openset.add(current)

    while len(openset) > 0:
        current = min(openset, key=lambda v: v.f)
        if current.position == goal:
            path = []
            while current.parent is not None:
                path.append(current)
                current = current.parent
            path.append(current)
            path.reverse()
            return path

        openset.remove(current)
        closedset.add(current)
        for node in children(current, current.position, grid):

            if node in closedset:
                continue

            new_g = current.g + 1
            if node.g > new_g:
                node.g = new_g
                node.parent = current
            else:
                node.g = new_g
                node.h = abs(node.position[0]-goal[0])+abs(node.position[1]-goal[1])
                node.f = node.g + node.h
                openset.add(node)

        #raise ValueError('No path found')


def closest_ghost(gameState):
    dist = gameState.data.ghostDistances
    alive = gameState.getLivingGhosts()[1:]

    closest_ghostt = (-1, -1)
    min_dist = 999999999999

    for i, ghost in enumerate(alive):
        if dist[i] < min_dist and alive[i]:
            min_dist = dist[i]
            closest_ghostt = gameState.getGhostPositions()[i]

    return closest_ghostt




def closest_food(gameState):

    minDistance = 900000
    pacmanPosition = gameState.getPacmanPosition()
    for i in range(gameState.data.layout.width):
        for j in range(gameState.data.layout.height):
            if gameState.hasFood(i, j):
                foodPosition = i, j
                distance = util.manhattanDistance(pacmanPosition, foodPosition)
                if distance < minDistance:
                    minDistance = distance
                    food = (i, j)
    return food





class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food

    ''' Print the layout'''
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print "---------------- TICK ", self.countActions, " --------------------------"
        # Dimensiones del mapa
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print "Width: ", width, " Height: ", height
        # Posicion del Pacman
        print "Pacman position: ", gameState.getPacmanPosition()
        # Acciones legales de pacman en la posicion actual
        print "Legal actions: ", gameState.getLegalPacmanActions()
        # Direccion de pacman
        print "Pacman direction: ", gameState.data.agentStates[0].getDirection()
        # Numero de fantasmas
        print "Number of ghosts: ", gameState.getNumAgents() - 1
        # Fantasmas que estan vivos (el indice 0 del array que se devuelve corresponde a pacman y siempre es false)
        print "Living ghosts: ", gameState.getLivingGhosts()
        # Posicion de los fantasmas
        print "Ghosts positions: ", gameState.getGhostPositions()
        # Direciones de los fantasmas
        print "Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)]
        # Distancia de manhattan a los fantasmas
        print "Ghosts distances: ", gameState.data.ghostDistances
        # Puntos de comida restantes
        print "Pac dots: ", gameState.getNumFood()
        # Distancia de manhattan a la comida mas cercada
        print "Distance nearest pac dots: ", gameState.getDistanceNearestFood()
        # Paredes del mapa
        print "Map:  \n", gameState.getWalls()
        # nearest food del mapa
        print "foooood:  \n", gameState.getFood()
        # Puntuacion
        print "Score: ", gameState.getScore()



    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        move = Directions.STOP
        legal = gameState.getLegalActions(0)

        goal = closest_ghost(gameState)
        start = gameState.getPacmanPosition()
        maze = gameState.getWalls()

        path = aStar(start, goal, maze)
        path_next = path[1].position

        next_move = path_next[0]-start[0], path_next[1]-start[1]

        if next_move == (0, -1):
            move = Directions.SOUTH
        elif next_move == (0, 1):
            move = Directions.NORTH
        elif next_move == (1, 0):
            move = Directions.EAST
        elif next_move == (-1, 0):
            move = Directions.WEST

        return move





    # def printLineData(self, state):
    #     """
    #     s = str(state.getGhostPositions()) + ", " + str(state.getGhostDirections()) + ", " \
    #            + str(state.getDistanceNearestFood)
    #     """
    #
    #     s = ", ".join([
    #         str(state.getPacmanPosition()),
    #         str(state.getGhostPositions()),
    #         str(state.data.ghostDistances)
    #
    #     ])
    #
    #     return s



class QLearningAgent(BustersAgent):

    def __init__(self, **args):
        "Initialize Q-values"
        BustersAgent.__init__(self, **args)

        self.actions = {"North": 0, "East": 1, "South": 2, "West": 3}
        self.table_file = open("qtable.txt", "r+")
        self.q_table = self.readQtable()
        self.epsilon = 0.0

        self.alpha = 0.25
        self.discount = 0.8 #gamma

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):

            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()

        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item) + " ")
            self.table_file.write("\n")

    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()

    def computePosition(self, tuplaState): #esto va a ser para mapear la tupla a cada row
        """
        Compute the row of the qtable for a given state.
        For instance, the state (3,1) is the row 7
        """
        if tuplaState[4] == 1:
            return 36
        elif tuplaState[4] == 2:
            return 37
        elif tuplaState[4] == 3:
            return 38
        elif tuplaState[4] == 4:
            return 39

        return tuplaState[0] + tuplaState[1] * 3 + tuplaState[2]*9 + tuplaState[3]*18


    def getQValue(self,gameState ,tuplaState, action):

        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(tuplaState)
        action_column = self.actions[action]

        return self.q_table[position][action_column]

    def computeValueFromQValues(self, gameState, tuplaState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """ 
        #returna el valor maximo de la row
        legalActions = gameState.getLegalActions(0)
        if len(legalActions) == 0:
            return 0
        return max(self.q_table[self.computePosition(tuplaState)])

    def computeActionFromQValues(self, gameState,tuplaState):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = gameState.getLegalActions(0)
        if "Stop" in legalActions:
            legalActions.remove("Stop")
        if len(legalActions) == 0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(gameState, tuplaState, legalActions[0])
        for action in legalActions:
            value = self.getQValue(gameState, tuplaState, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def chooseAction(self, gameState):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        tuplaState = self.printLineData(gameState)
        # Pick Action
        legalActions = gameState.getLegalActions(0)
        if "Stop" in legalActions:
            legalActions.remove("Stop")

        action = None

        if len(legalActions) == 0:
            return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(gameState,tuplaState)

    def update(self, gameState, tuplaState, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
              Good Terminal state -> reward 1
              Bad Terminal state -> reward -1
              Otherwise -> reward 0
              Q-Learning update:
              if terminal_state:
                Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
              else:
                Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))
        """
        "*** YOUR CODE HERE ***"
        position = self.computePosition(tuplaState) #selects row from qtable
        action_num = self.actions.get(action) #selects column
        legalActions = gameState.getLegalActions(0)

        if len(legalActions) == 1:
            self.q_table[position][action_num] = (1 - self.alpha) * self.q_table[position][action_num] + \
                                                 self.alpha * (reward + 0)
        else:
            self.q_table[position][action_num] = (1 - self.alpha) * self.q_table[position][action_num] + \
                                                 self.alpha * (reward + self.discount * self.computeValueFromQValues(gameState,nextState))




    def getPolicy(self, gameState,tuplaState):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(gameState,tuplaState)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)


    def printLineData(self, state):
        distancer = distanceCalculator.Distancer(state.data.layout)
        #here we create NearestH and NearestV and free move
        free_move = 0
        if state.getNumFood() > 0:
            minDistance = 900000
            pacmanPosition = state.getPacmanPosition()
            for i in range(state.data.layout.width):
                for j in range(state.data.layout.height):
                    if state.hasFood(i, j):
                        foodPosition = i, j

                        # distlayout = distanceCalculator.computeDistances(state.data.layout)
                        # distance = distanceCalculator.getDistanceOnGrid(distlayout, state.getPacmanPosition(), (i, j))
                        distance = distancer.getDistance(state.getPacmanPosition(), (i, j))

                        if distance < minDistance:
                            NearestH = i-pacmanPosition[0]
                            NearestV = j-pacmanPosition[1]

            if NearestV == 0:
                pos = range(1,abs(NearestH))
                sign = 1
                if NearestH < 0:
                    sign *= -1
                pos = [i*sign for i in pos]
                check_pos = 0
                for i in pos:
                    check_pos = state.hasWall(pacmanPosition[0]+i, pacmanPosition[1])
                    if check_pos:
                        free_move = 0
                        break
                    else:
                        free_move = 2+1*sign #1 for left, 3 for right
            elif NearestH == 0:
                pos = range(1,abs(NearestV))
                sign = 1
                if NearestV < 0:
                    sign *= -1
                pos = [i*sign for i in pos]
                check_pos = 0
                for i in pos:
                    check_pos = state.hasWall(pacmanPosition[0], pacmanPosition[1]+i)
                    if check_pos:
                        free_move = 0
                        break
                    else:
                        free_move = 3+1*sign #2 for down, 4 for up

        else: #for ghosts
            NearestH = closest_ghost(state)[0] - state.getPacmanPosition()[0]
            NearestV = closest_ghost(state)[1] - state.getPacmanPosition()[1]

            if NearestV == 0:
                pos = range(1, abs(NearestH))
                sign = 1
                if NearestH < 0:
                    sign *= -1
                pos = [i*sign for i in pos]
                check_pos = 0
                for i in pos:
                    check_pos = state.hasWall(state.getPacmanPosition()[0]+i, state.getPacmanPosition()[1])
                    if check_pos:
                        free_move = 0
                        break
                    else:
                        free_move = 2+1*sign #1 for left, 3 for right
            elif NearestH == 0:
                pos = range(1, abs(NearestV))

                sign = 1
                if NearestV < 0:
                    sign *= -1
                pos = [i*sign for i in pos]
                check_pos = 0
                for i in pos:
                    check_pos = state.hasWall(state.getPacmanPosition()[0], state.getPacmanPosition()[1]+i)


                    if check_pos:
                        free_move = 0
                        break
                    else:
                        free_move = 3+1*sign #2 for down, 4 for up



        #Here we create hor and ver
        if NearestH > 0:
            NearestH = 1 #to the right
        elif NearestH < 0:
            NearestH = 2 #to the left
        else:
            NearestH = 0 #exact axis

        if NearestV > 0:
            NearestV = 1 #up
        elif NearestV < 0:
            NearestV = 2 #down
        else:
            NearestV = 0 #exact axis


        hor = 0
        if NearestH == 1:
            if "East" in state.getLegalPacmanActions():
                hor = 1
        elif NearestH == 2:
            if "West" in state.getLegalPacmanActions():
                hor = 1
        ver = 0
        if NearestV == 1:
            if "North" in state.getLegalPacmanActions():
                ver = 1
        elif NearestV == 2:
            if "South" in state.getLegalPacmanActions():
                ver = 1

        tupla_state = (NearestH, NearestV, hor, ver, free_move)

        return tupla_state


    def scorito(self,state):
        return state.getScore()

