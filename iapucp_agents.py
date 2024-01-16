from game import Directions
from game import Agent
from game import Actions

import random

import util
import searchAgents

#Importando librerías para ML
# requiere haber instalado (sugerencia: usar pip): scipy, numpy, matplotlib, pandas, 
# sklearn, keras, tensorflow

# import warning en tf: https://stackoverflow.com/questions/66092421/how-to-rebuild-tensorflow-with-the-compiler-flags
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# version de Python
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy as np
print('numpy: {}'.format(np.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas as pd
print('pandas: {}'.format(pd.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
# keras
import tensorflow.keras as keras
print('keras: {}'.format(keras.__version__))
# pickle
import pickle
print('pickle: {}'.format(pickle.format_version))

# Fin de importación de librerías

# no olvidar actualizar la variable N!
N=10 # el vector de atributos extra, para el código de ejemplo, tiene 2 elementos
cantFeatures = 10 + N

# FUNCIÓN PARA OBTENER FEATURES A PARTIR DE UN GAMESTATE
def obtenerFeatures(gState, agregarExtraAttributes=True):
    """
    Esta es la función que obtiene los features a partir de un GameState del entorno.
    Para el desafío, no es necesario modificar esta función, pero si desea, puede hacerlo, para obtener mejores features para el aprendizaje.
    Características extraidas [12]: Posicion en x del ghost1 respecto
    a pacman, Posicion en y del ghost1 respecto a pacman, Posicion en x del ghost2 respecto a pacman,
    Posicion en y del ghost2 respecto a pacman, Cantidad de cápsulas restantes,
    Distancia manhattan al fantasmita más cercano, Distancia manhattan a la cápsula más cercana,
    Promedio de las distancias manhattan de las 5 comidas más cercanas (no capsulas), Score,
    Cantidad de fantasmas asustados
    """
    features = np.array([])
    #gState_successor = gState.generateSuccessor(0, accion)
    #isWin (1/0), isLose (1/0)
    #features = np.append(features, [ int(gState_successor.isWin()) , int(gState_successor.isLose()) ])

    pac_pos = gState.getPacmanPosition()
    ghosts_poss = gState.getGhostPositions()

    ghosts_poss_relToPacman = np.array([np.array(x) - np.array(pac_pos) for x in ghosts_poss]).astype(int)

    #print("pacposs", pac_pos)
    #print("fantasmiposs", ghosts_poss)

    #print("relativePos", ghosts_poss_relToPacman)

    features = np.append(features, ghosts_poss_relToPacman)
    
    capsules = gState.getCapsules()

    # Feature de cantidad de capsulas
    features = np.append(features, len(capsules))

    state_food = gState.getFood()
    food = [(x, y) #enlista las posiciones donde hay comida
            for x, row in enumerate(state_food)
            for y, food in enumerate(row)
            if food]
    nearest_ghosts = sorted([util.manhattanDistance(pac_pos, i) for i in ghosts_poss])

    # Feature de Fantasmita Mas Cercano: a cuanta distancia manhattan esta el fantasma mas cercano
    features = np.append(features, [ nearest_ghosts[0] ])
    ############################lo de arriba esta bien
    # Feature de Pildora mas cercana #a cuanta distancia manhattan esta la capsula mas cercana
    nearest_caps = sorted([util.manhattanDistance(pac_pos, i) for i in capsules])
    if nearest_caps:
        manhDist_nearestCaps = nearest_caps[0]
    else:
        manhDist_nearestCaps = max(gState.data.layout.width,gState.data.layout.height)
    features = np.append(features, [manhDist_nearestCaps])
    # Feature del promedio de MD a las 5 comidas mas cercanas. Que pasa cuando hay menos de 5?
    nearest_food = sorted([(util.manhattanDistance(pac_pos, i),i) for i in food])
    nearest_food = nearest_food[:5]
    for i in range(min(len(nearest_food), 5)):
        nearest_food[i]=searchAgents.mazeDistance(pac_pos,nearest_food[i][1],gState)

    features = np.append(features, sum(nearest_food)/len(nearest_food))

    # Feature de Score
    features = np.append(features, [gState.getScore()] )

    # Feature de cantidad de Fantasmitas Asustaditos
    ghostStates = gState.getGhostStates()
    numOfScaredGhosts = 0
    for ghostState in ghostStates:
        if ghostState.scaredTimer > 0:
            numOfScaredGhosts += 1

    features = np.append(features, [numOfScaredGhosts] )

    ####### Agregamos un vector N-dimensional con los N atributos extra
    
    # Leyenda de los N=4 elementos que vamos a agregar
    # en el código de ejemplo se agregarán N=4 elementos
    # (agregaremos un elemento por cada una de las 4 acciones fisicas de mov.)
    # 1: no wall (dirección válida para moverse)
    # 2: wall

    # Así se obtiene la lista con las acciones legales, a partir del state actual
    legalActions = gState.getLegalActions()

    #print("Acciones legales:", legalActions)

    lista_newAttributes = []
    lista_all_relevant_moves = ['West', 'East', 'North', 'South']

    #print(f'pos_orig =============== {pac_pos}')
    for ii in range(0,4): # N=4, ya que en este ejemplo agregamos una dimension por cada dirección de mov.
        # si no es una acción legal, es porque en esa dirección hay un wall
        if lista_all_relevant_moves[ii] in legalActions:
            dir = (0,0)
            if lista_all_relevant_moves[ii] == 'West':        
                dir = (-1,0)
            elif lista_all_relevant_moves[ii] == 'East':        
                dir = (1,0)
            elif lista_all_relevant_moves[ii] == 'North':        
                dir = (0,1)
            elif lista_all_relevant_moves[ii] == 'South':        
                dir =  (0,-1)
            
            newPos = (pac_pos[0] + dir[0], pac_pos[1] + dir[1])
            #print(f'new_pos = {lista_all_relevant_moves[ii]}: {newPos}')
            if newPos in ghosts_poss:
                feature_movil = 1 # fantasma
            elif newPos in food:
                feature_movil = 4 # comestible
            elif newPos in capsules:
                feature_movil = 4 # comestible
            else:
                feature_movil = 3 # otro, vacío
        else:
            feature_movil = 2 # muro

        #print(f'new_feature = {lista_all_relevant_moves[ii]}: {feature_movil}')
        lista_newAttributes.append(feature_movil)
    
    ####### Features propias:
    centro = (9,5)
    limiteIzq = 5
    limiteDer = 14
    limiteCentro = 5
    ## posx_pacman_fromCenter
    lista_newAttributes.append(pac_pos[0]-centro[0])
    ## posy_pacman_fromCenter
    lista_newAttributes.append(pac_pos[1]-centro[1])
    #print(f'x: {pac_pos[0]-centro[0]}, y: {pac_pos[1]-centro[1]}')
    
    ##cant_lados
    lado1 = 0
    lado2 = 0
    lado3 = 0
    lado4 = 0
    for dot in food:
        if dot[0] < limiteIzq:
            lado1 += 1
        elif dot[0] > limiteDer:
            lado2 += 1
        elif dot[1] > limiteCentro:
            lado3 += 1
        elif dot[1] < limiteCentro:
            lado4 +=1
    
    #print(f'1: {lado1}, 2: {lado2}, 3: {lado3}, 4:{lado4}')
    lista_newAttributes.append(lado1)
    lista_newAttributes.append(lado2)
    lista_newAttributes.append(lado3)
    lista_newAttributes.append(lado4)
    
    if agregarExtraAttributes: features = np.append(features,  lista_newAttributes )
    
    #print(features) # en este código de ejemplo, el único valor decimal (non integer) es el
    # que corresponde al Promedio de las dists manhattan de las 5 comidas más cercanas

    return features

# CLASE QUE IMPLEMENTA UN AGENTE ALEATORIO PERSONALIZABLE
class my_Random_Agent(Agent):
    """
    This is a RANDOM agent!
    """

    def __init__(self):

        print()
        print("="*15)
        print()
        print("Se inicializó el agente ALEATORIO")
        print("="*15)
   

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        features = obtenerFeatures(state).reshape(1,-1)

        # Así se obtiene la lista con las acciones legales, a partir del state actual
        legalActions = state.getLegalActions()

        return legalActions[  random.randint(1,len(legalActions))-1 ]

# CLASE QUE IMPLEMENTA EL AGENTE BASADO EN BEHAVIORAL CLONING
class my_ML_Agent(Agent):
    """
    This is a behaviour clonned agent!
    """

    def __init__(self):
        import pickle
        import numpy as np

        # open a file, where you stored the pickled data
        file_modeloCargado = open('modeloEntrenado.p', 'rb')
        file_scalerCargado = open('scaler.p','rb')

        # load information from that file
        self.modelo = pickle.load(file_modeloCargado)
        self.scaler = pickle.load(file_scalerCargado)
        self.stops_seguidos = 0

        # close the file
        file_modeloCargado.close()
        file_scalerCargado.close()

        self.cantAccionesInvalidas = 0

        print()
        print("="*15)
        print()
        print("Se inicializó el agente basado en Behavior Cloning")
        print("Autor: Grupo número 3\n")
        print("="*15)
   

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        features = obtenerFeatures(state).reshape(1,-1)
        X = self.scaler.transform(features)

        #Si es un DecisionTreeClassifier o un RandomForestClassifier
        #accionNum = int(self.modelo.predict(X))
        probas = self.modelo.predict_proba(X)[0] #obtenemos el arreglo con la probabilidad por acción
        accionList = np.flip(np.argsort(probas)) #ordenamos los índices de mayor a menor
        print(f'acciones {accionList}')
        maxStops = 10 # colocamos un tope a los stops seguidos

        #Si es un keras sequential
        #accionNum = self.modelo.predict(features).argmax(axis=-1)

        #Convertir el índice de la acción a su respectivo string
        movelist = ['Stop','East','North','West','South']

        # Así se obtiene la lista con las acciones legales, a partir del state actual
        legalActions = state.getLegalActions()
        ####Si deseas, usa la variable `self.cantAccionesInvalidas`

        #Codificar el comportamiento del agente para el caso de predecir una accion inválida
        accionNum = 0 # acción por defecto
        for accion in accionList:
            if movelist[accion] in legalActions:
                accionNum = accion
                # validar que no haga más de 10 stops seguidos
                if accionNum == 0:
                    self.stops_seguidos = self.stops_seguidos + 1
                else:
                    self.stops_seguidos = 0
                
                if(self.stops_seguidos > maxStops):
                    continue

                print(f'elegida {accionNum}')
                break

        return movelist[accionNum]