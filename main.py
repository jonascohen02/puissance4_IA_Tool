import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 
import numpy as np
import random
import tensorflow as tf
from Dqn import DQNAgent
from Env import Game
from functions import play_episode
# Disable absl INFO and WARNING log messages
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)


counter_game = 0

def play_against(agent,train=False,n=1):

    global counter_game
    game.reset()
    # print("\n \n Agent ",agent.session,": accuracy",agent.stats["accuracy"],"\n")
    for episode in range(n):
        game.reset()
        previous = {
            "state":'',
            "action" :'',
        }
        state = np.array(game.grid).reshape((1, game.height, game.width))
        
        # IA qui commence, 1 sinon humain commence et 0 donc p%2 = humain
        p = 1 if random.random() > 0.5 else 0
        while game.winner is None:
            # Si l'humain joue p=0
            if p%2 == 0:
                # On Vérifie que l'entrée est un nombre et est autorisée
                # os.system("cls")
                game.display_grid()
                userAction = input("Choose a column (1-7): ")
                while not userAction.isdigit() or int(userAction)-1 not in game.legal_actions:
                    # os.system("cls")
                    game.display_grid()
                    userAction = input("Invalid choice, retry (1-7): ")
                action = int(userAction)-1     
            else:
                # Action, 1 step:
                np_state = np.array(state).reshape((1, game.height, game.width))
                predictAction = agent.get_action(np_state)
                while not predictAction[0]:
                    agent.update([game.grid], [predictAction[1]], [-20], [game.grid], epochs=2, verbose=0)
                    predictAction = agent.get_action(np_state)
                action = predictAction[1]
                previous["action"] = action
                previous["state"] = state

            winner,reward = game.add_piece(action)
            new_state = np.array(game.grid).reshape((1, game.height, game.width))

            if winner is None:
                # Pas de gagnant, on continue et reward de 0
                r=0
                forceUpdate = False
            elif winner =="egal":
                # Egalite reward de 0
                r=0
                agent.stats["egalites"] += 1
                # os.system("cls")
                game.display_grid()
                print("Égalité, pas de vainqueurs")
                forceUpdate = True
            else:
                # Victoire d'un joueur, si p%2=0, l'humain gagne donc reward de -1 sinon 1
                r = -1 if p%2 == 0 else 1
                agent.stats["wins"]+=p%2 #Update stats, +1 victoire (car p%2 sera égale à 1) si il gagne, sinon 0
                # os.system("cls")
                game.display_grid()
                print("Player ", winner, "wins! \n")
                forceUpdate = True

            # On entraine l'IA tous les deux coups sauf au premier coup car le new_state correspond à l'état après l'action de l'IA + celle du joueur sauf dernier coup:
            # forceUpdate est True si il n'y a plus de coup après, dans le cas ou l'IA gagne ou fait une egalite. Ainsi on attend plus le coup de l'adversaire
            if train and (p%2==0 and p!=0 or forceUpdate):
                # print("\nPrevious state: ", previous["state"],
                # '\nPrevious: ',previous["action"],
                # '\nReward: ',r,
                # '\nNew state: ',new_state,
                # '\nGame winner: ',game.winner,
                # '\nP%2 =',p%2,"\n\n"
                # )
                agent.update(previous["state"], previous["action"], r, new_state, game.legal_actions)
            
            # if forceUpdate:
            #     break
            p+=1
            state = new_state
        counter_game +=1
        agent.stats["games"]+=1
    if train:
        agent.model_save("against-human/")
    agent.display_stats()




if __name__ == "__main__":
    # Initialisation du jeu et de l'agent
    os.system("cls")
    game = Game(6, 7)
    # Voici la grille ainsi créé, une matrice de 6x7
    # grid= [
    #     [0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0],
    #     [0,2,1,0,0,0,0]
    # ]


    # Création de 2 agents:
    agent1 = DQNAgent(game,session=0, epsilon=0.8,learning_rate=0.01)
    agent2 = DQNAgent(game,session=1, epsilon=0.8,learning_rate=0.01)
    # agent2 = DQNAgent(game,session=1, epsilon=0.8,learning_rate=0.01, loadModel='ia2/')    # 0.9
    # play_against(agent2,False,15)
    os.system("cls")
    play_against(DQNAgent(game,session=1, epsilon=0,learning_rate=0.01, loadModel="ia1/"),train=False,n=5)
    for i in range(10):
        newEps = max(0.1,(1-i)/10)
        print("\n\nNew Epsilon = ",newEps," \n\n")
        agent1.set_epsilon(newEps)
        agent2.set_epsilon(newEps)
        play_episode(agent1,agent2,1000,20, game, maxSize=10000)
        agent1.display_stats()
        agent2.display_stats()




# ATTENTION INVERSE GAME DANS PLAY EPISODE


    # for i in range(10):
        # newEps = max(0.1,(1-i)/10)
        # print("\n\nNew Epsilon = ",newEps," \n\n")
        # agent1.set_epsilon(newEps)
        # agent2.set_epsilon(newEps)
        # play_episode(agent1,agent2,1000,20, game, maxSize=10000)
        # agent1.display_stats()
        # agent2.display_stats()