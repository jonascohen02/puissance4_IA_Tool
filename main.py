import os
# Remove useless informations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 
from Dqn import DQNAgent
from Env import Game
from functions import play_episode, play_against

if __name__ == "__main__":
    # On vide la console
    os.system("cls")


    # Initialisation du jeu et de l'agent
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

    # On pourra dans des scripts futurs recharger nos agents:
    # agent1 = DQNAgent(game,session=0,loadModel='firstAgent/', epsilon=0.8,learning_rate=0.01)
    # agent2 = DQNAgent(game,session=1,loadModel='secondAgent/' epsilon=0.8,learning_rate=0.01)
    # Permet de jouer contre son agent
    # play_against(agent1,game,numberEpisodes=5,train=True,trainRate=5,maxSize=10)



    # Création de 2 agents:
    agent1 = DQNAgent(game,session=0, epsilon=0.8,learning_rate=0.01)
    agent2 = DQNAgent(game,session=1, epsilon=0.8,learning_rate=0.01)

    # Exemple d'utilisations:

    # On va faire 10 séries de 1000 épisodes et on va à chaque série réduire un peu epsilon pour atteindre 0.1. 
    # On affiche les stats des agents à chaque fin dé série
    for i in range(10):
        newEps = max(0.1,(1-i)/10)
        print("\n\nNew Epsilon = ",newEps," \n\n")
        agent1.set_epsilon(newEps)
        agent2.set_epsilon(newEps)
        play_episode(agent1,agent2,game,numberEpisodes=1000,trainRate=20, maxSize=10000)
        agent1.display_stats()
        agent2.display_stats()

    # On sauvegarde nos modèles:
    agent1.model_save('firstAgent/')
    agent2.model_save('secondAgent/')


    # On teste nos agents sans les entraîner sur 1 partie pour chaque agent
    play_against(agent1,game,numberEpisodes=1)
    play_against(agent2,game,numberEpisodes=1)



