import numpy as np
import random
import tensorflow as tf
import os
from copy import deepcopy
# Première étape, à tester sur 1 épisode pour vérifier le tableau en sortie 
# + Vérifier en cas d'égalité ! 

# TO DO: 
# Ajouter une fonctionnalité pour lancer les états en mode aléatoire sans forcément du début
# Verifier si y'a blocage, si oui, +1


states = [[],[]]
actions = [[],[]]
rewards = [[],[]]
nextstates=[[],[]]

def play_episode(p1,p2,numberEpisodes, trainRate, game, maxSize=10000):
    global counter_game
    players = [p1, p2]
    # Boucle d'entraînement
    global states, actions, rewards, nextstates
    # Sauvegarde de tous les etats pour les  2 agents

    episode = 0
    counter = 1
    while episode < numberEpisodes:
        p=0
        p1.stats["games"] += 1
        p2.stats["games"] += 1
        game.reset()
        random.shuffle(players)
        game.player = players[0].session+1
        while game.winner is None:
            randomIndex = random.random()
            indexPlayer = players[p%2].session
            indexNextPlayer = players[(p+1)%2].session

            state = deepcopy(game.grid)        #Eviter que lors de la modification, tous les states se modifient     
            np_state = np.array(state).reshape((1, game.height, game.width))
            

            # Action, 1 step:
            predictAction = players[p%2].get_action(np_state)
            
            # Si l'action n'est pas autorisée, reward négatif et on redemande
            while not predictAction[0]:
                tempRandom = random.random() 
                # On consière ça comme un tour de jeu avec une reward négative et l'état n'a pas changé
                states[indexPlayer].insert(round(tempRandom*len(states[indexPlayer])),state)
                actions[indexPlayer].insert(round(tempRandom*len(states[indexPlayer])), predictAction[1])
                nextstates[indexPlayer].insert(round(tempRandom*len(states[indexPlayer])),state)
                rewards[indexPlayer].insert(round(tempRandom*len(states[indexPlayer])), -20)
                # Realiser une update pour ne pas boucler
                players[p%2].update([state], [predictAction[1]], [-20], [state], epochs=2, verbose=0)
                # On redemande de nous prédire
                predictAction = players[p%2].get_action(np_state)

            action = predictAction[1]

            # Ajout dans la grande dataset du State du joueur en cours avant de jouer dans une position aléatoire
            states[indexPlayer].insert(round(randomIndex*len(states[indexPlayer])),state)
           
            # Envoie de l'action au jeu
            winner, rewardPlayer = game.add_piece(action)

            # On insert cette action dans le dataset au même index que le state
            actions[indexPlayer].insert(round(randomIndex*len(actions[indexPlayer])),action)
            
            # Sauvegarde de l'état actuel comme étant l'état suivant le coup du joueur précédent (adversaire, aussi celui qui joue après) 
            new_state = deepcopy(game.grid)

            # On ajoute cet état au même index que les précédents states actions seulement si p!=0 car l'adversaire n'a pas encore joué
            if p!=0:
                nextstates[indexNextPlayer].insert(round(prevRandomIndex*len(nextstates[indexNextPlayer])),new_state)


            # Si égalité:
            if winner == "egal":
                rewards[indexPlayer].insert(round(randomIndex*len(rewards[0])),rewardPlayer+1)
                rewards[indexNextPlayer].insert(round(prevRandomIndex*len(rewards[1])),rewardPlayer+1)
                nextstates[indexPlayer].insert(round(randomIndex*len(nextstates[indexPlayer])),new_state)
                p1.stats["egalites"] += 1
                p2.stats["egalites"] += 1
                break

            # Si p%2 a gagné:
            elif winner is not None:
                rewards[indexPlayer].insert(round(randomIndex*len(rewards[indexPlayer])),rewardPlayer+15)
                rewards[indexNextPlayer].insert(round(prevRandomIndex*len(rewards[indexNextPlayer])),rewardPlayer-15)
                nextstates[indexPlayer].insert(round(randomIndex*len(nextstates[indexPlayer])),new_state)
                players[p%2].stats["wins"] += 1
                break

            # Ni p%2 ni (p+1)%2 n'a gagné (ce dernier n'a pas joué), donc coup suivant
            else:
                if p!=0:
                    rewards[indexNextPlayer].insert(round(prevRandomIndex*len(rewards[indexNextPlayer])),rewardPlayer+0)
            p +=1
            prevRandomIndex = randomIndex
        

        # if (counter)%100 == 0:
        #     print("Partie n°{}".format(counter))
        #     print("Player 1 Rows length: ",len(states[0]))
        #     print("Player 2 Rows length: ",len(states[1]))

        # Tous les x parties, on train nos modeles
        if (episode+1)%trainRate == 0 and len(states[indexPlayer]) > (maxSize-trainRate*21):
            print("\n\n\nEpisode n°", episode+1)
            print("\n\nAgent1:")
            print("Length data: ",len(states[0])," rows\n")
            p1.update(states[p1.session],actions[p1.session],rewards[p1.session],nextstates[p1.session])
            print("\n\nAgent2:")
            print("Length data: ",len(states[1])," rows\n")
            p2.update(states[p2.session],actions[p2.session],rewards[p2.session],nextstates[p2.session])
            p1.display_stats(rewards[0])
            p2.display_stats(rewards[1])
            p1.model_save("ia1/")
            p2.model_save("ia2/")


        # Si plus de 10000 éléments dans le tableau, on retire le dernier 
        if len(states[indexPlayer]) > maxSize:
            states[indexPlayer] = states[indexPlayer][1:maxSize]
            nextstates[indexPlayer] = nextstates[indexPlayer][1:maxSize]
            actions[indexPlayer] = actions[indexPlayer][1:maxSize]
            rewards[indexPlayer] = rewards[indexPlayer][1:maxSize]
    
        if(len(states[indexPlayer]) > (maxSize-trainRate*21)):
            # Débuter les entraînements à partir du moment ou l'ensemble est plein
            episode += 1
        
        counter += 1


def play_against(p1, game, train=False, numberEpisodes=1, trainRate=None, maxSize=10000):
    global counter_game
    # Boucle d'entraînement
    global states, actions, rewards, nextstates
    # Sauvegarde de tous les etats pour les  2 agents

    episode = 0
    counter = 1
    while episode < numberEpisodes:
        p = 1 if random.random() > 0.5 else 0
        p1.stats["games"] += 1
        game.reset()

        while game.winner is None:
            randomIndex = random.random()
            state = deepcopy(game.grid) 
            if p%2 == 0:   
                game.display_grid()
                userAction = input("Choose a column (1-7): ")
                while not userAction.isdigit() or int(userAction)-1 not in game.legal_actions:
                    # os.system("cls")
                    game.display_grid()
                    userAction = input("Invalid choice, retry (1-7): ")
                action = int(userAction)-1
            else: 
                np_state = np.array(state).reshape((1, game.height, game.width))
                predictAction = p1.get_action(np_state)
                while not predictAction[0]:
                    tempRandom = random.random() 
                    predictAction = p1.get_action(np_state)
                    states[0].insert(round(tempRandom*len(states[0])),state)
                    actions[0].insert(round(tempRandom*len(states[0])), predictAction[1])
                    nextstates[0].insert(round(tempRandom*len(states[0])),state)
                    rewards[0].insert(round(tempRandom*len(states[0])), -20)
                    
                    p1.update([game.grid], [predictAction[1]], [-20], [game.grid], epochs=2, verbose=0)

                action = predictAction[1]
                # randomIndex previous["action"] = action
                # randomIndex previous["state"] = state
            
            winner,reward = game.add_piece(action)
            new_state = np.array(game.grid).reshape((1, game.height, game.width))



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

