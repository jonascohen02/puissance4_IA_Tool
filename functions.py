import numpy as np
import random
from copy import deepcopy


states = [[],[]]
actions = [[],[]]
rewards = [[],[]]
nextstates=[[],[]]

def play_episode(p1,p2,game, numberEpisodes, trainRate, maxSize=10000):
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
        # On démarre le jeu au hasard
        random.shuffle(players)
        game.player = players[0].session+1
        
        # A chaque coup:
        while game.winner is None:
            # Utilisé pour insérer chaque ligne de l'échantillon dans un ordre aléatoire
            randomIndex = random.random()
            # Index du joueur en cours
            indexPlayer = players[p%2].session
            # Index de l'adversaire
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

            # On ajoute cet état au même index que les précédents states actions seulement si p!=0 car sinon l'adversaire n'a pas encore joué
            if p!=0:
                nextstates[indexNextPlayer].insert(round(prevRandomIndex*len(nextstates[indexNextPlayer])),new_state)


            # Si égalité:
            if winner == "egal":
                # On affecte les récompenses de 1 pour tous et on modifie les stats
                # On ajoute aussi l'état actuel à l'état suivant de l'adversaire
                rewards[indexPlayer].insert(round(randomIndex*len(rewards[0])),rewardPlayer+1)
                rewards[indexNextPlayer].insert(round(prevRandomIndex*len(rewards[1])),rewardPlayer+1)
                nextstates[indexPlayer].insert(round(randomIndex*len(nextstates[indexPlayer])),new_state)
                p1.stats["egalites"] += 1
                p2.stats["egalites"] += 1
                break

            # Si p%2 a gagné:
            elif winner is not None:
                # On affecte les récompenses de 15 à celui qui vient de jouer (et donc de gagner) et -15 à l'adversaire et on modifie les stats
                # On ajoute aussi l'état actuel à l'état suivant de l'adversaire
                rewards[indexPlayer].insert(round(randomIndex*len(rewards[indexPlayer])),rewardPlayer+15)
                rewards[indexNextPlayer].insert(round(prevRandomIndex*len(rewards[indexNextPlayer])),rewardPlayer-15)
                nextstates[indexPlayer].insert(round(randomIndex*len(nextstates[indexPlayer])),new_state)
                players[p%2].stats["wins"] += 1
                break

            # Ni p%2 ni (p+1)%2 n'a gagné (ce dernier n'a pas joué), donc coup suivant et on ajoute la récompense de 0 à l'adversaire
            else:
                if p!=0:
                    rewards[indexNextPlayer].insert(round(prevRandomIndex*len(rewards[indexNextPlayer])),rewardPlayer+0)
            p +=1
            # On sauvegarde le randomIndex pour pouvoir sauvegarder les récompenses et next_state du joueur actuel
            prevRandomIndex = randomIndex
        

        # Tous les x parties, on train nos modeles
        if (episode+1)%trainRate == 0 and len(states[indexPlayer]) > (maxSize-trainRate*21):
            # On affiche des informations
            # On lance la fonction qui train automatiquement
            print("\n\n\nEpisode n°", episode+1)
            print("\n\nAgent1:")
            print("Length data: ",len(states[0])," rows\n")
            p1.update(states[p1.session],actions[p1.session],rewards[p1.session],nextstates[p1.session])
            print("\n\nAgent2:")
            print("Length data: ",len(states[1])," rows\n")
            p2.update(states[p2.session],actions[p2.session],rewards[p2.session],nextstates[p2.session])

            # On affiche les stats
            p1.display_stats(rewards[0])
            p2.display_stats(rewards[1])

            # On fait une sauvegarde des modèles
            p1.model_save("ia1/")
            p2.model_save("ia2/")


        # Si plus de 10000 éléments dans le tableau, on retire le dernier 
        if len(states[indexPlayer]) > maxSize:
            states[indexPlayer] = states[indexPlayer][1:maxSize]
            nextstates[indexPlayer] = nextstates[indexPlayer][1:maxSize]
            actions[indexPlayer] = actions[indexPlayer][1:maxSize]
            rewards[indexPlayer] = rewards[indexPlayer][1:maxSize]
    
        # Débuter les entraînements à partir du moment ou l'ensemble est plein
        if(len(states[indexPlayer]) > (maxSize-trainRate*21)):
            episode += 1
            
        # Affiche un message toutes les 10 parties
        # if counter%10 == 10:
        print("\n\n\nPartie n°", counter+1)
        print("Length data: ", len(states[0])," rows\n")
        counter += 1


def play_against(p1, game, train=False, numberEpisodes=1, trainRate=None, maxSize=10000):
    global counter_game
    # Boucle d'entraînement
    global states, actions, rewards, nextstates
    # Sauvegarde de tous les etats pour les  2 agents

    episode = 0
    counter = 1

    previous = {
        "state":'',
        "action" :'',
        "reward":''
    }
    while episode < numberEpisodes:
        p = 1 if random.random() > 0.5 else 0
        p1.stats["games"] += 1
        # On reset le jeu
        game.reset()
        game.display_grid()
        # Tant que le jeu n'est pas fini, à chaque coup:
        while game.winner is None:
            # Utilisé pour insérer chaque ligne de l'échantillon dans un ordre aléatoire
            randomIndex = random.random()
            # Eviter que lors de la modification de la grille, tous les states se modifient     
            state = deepcopy(game.grid) 

            # Détermine si le joueur actuel est humain ou pas
            if p%2 == 0:   
                # On affiche la grille et demande au joueur une colonne tant que la réponse n'est pas correcte
                userAction = input("Player {}:\nChoose a column (1-7): ".format(game.player))
                while not userAction.isdigit() or int(userAction)-1 not in game.legal_actions:
                    game.display_grid()
                    userAction = input("Player {}:\nInvalid choice, retry (1-7): ".format(game.player))
                # les vraies index des colonnes sont de 0 à 6
                action = int(userAction)-1
            
            else: 
                # On convertit la grille de liste en matrice
                np_state = np.array(state).reshape((1, game.height, game.width))

                # On prédit une action, si l'action n'est pas autorisée, reward négatif, update pour ne pas boucler à l'infini
                # et on redemande
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
                print("Player {}:\nAction choosen: {}".format(game.player,action+1))
                if train:
                    previous["action"] = action
                    previous["state"] = state
            
            # Envoie de l'action au jeu
            winner, reward = game.add_piece(action)
            print("\n")
            game.display_grid()
            if p%2 == 1 and train:
                previous["reward"] = reward
            new_state = deepcopy(game.grid)

            # Si égalité:
            if winner == "egal":
                print("\n\n\nEgalites, no winner ! \n\n\n")
                # On affecte les récompenses de 1 pour tous et on modifie les stats
                # On ajoute aussi l'état actuel à l'état suivant de l'adversaire
                if train:
                    if p%2 == 0:
                        rewards[0].insert(round(randomIndex*len(rewards[0])),previous["reward"]+1)
                        nextstates[0].insert(round(randomIndex*len(nextstates[0])),new_state)
                        states[0].insert(round(randomIndex*len(nextstates[0])),previous["state"])
                        actions[0].insert(round(randomIndex*len(nextstates[0])),previous["action"])
                    else:
                        rewards[0].insert(round(randomIndex*len(rewards[0])),previous["reward"]+1)
                        nextstates[0].insert(round(randomIndex*len(nextstates[0])),new_state)
                        states[0].insert(round(randomIndex*len(nextstates[0])),previous["state"])
                        actions[0].insert(round(randomIndex*len(nextstates[0])),previous["action"])
                p1.stats["egalites"] += 1
                break

            # Si p%2 a gagné:
            elif winner is not None:
                print("\n\n\nWinner: Player {} \n\n\n".format(game.winner))
                if train:
                    if p%2 == 0:
                        rewards[0].insert(round(randomIndex*len(rewards[0])),previous["reward"]-15)
                        nextstates[0].insert(round(randomIndex*len(nextstates[0])),new_state)
                        states[0].insert(round(randomIndex*len(nextstates[0])),previous["state"])
                        actions[0].insert(round(randomIndex*len(nextstates[0])),previous["action"])
                    else:
                        rewards[0].insert(round(randomIndex*len(rewards[0])),previous["reward"]+15)
                        nextstates[0].insert(round(randomIndex*len(nextstates[0])),new_state)
                        states[0].insert(round(randomIndex*len(nextstates[0])),previous["state"])
                        actions[0].insert(round(randomIndex*len(nextstates[0])),previous["action"])
                        p1.stats["wins"] += 1
                break

            # Ni p%2 ni (p+1)%2 n'a gagné (ce dernier n'a pas joué), donc coup suivant et on ajoute la récompense de 0 à l'adversaire
            else:
                if train and p%2 == 0 and p!=0:
                    rewards[0].insert(round(randomIndex*len(rewards[0])),previous["reward"])
                    nextstates[0].insert(round(randomIndex*len(nextstates[0])),new_state)
                    states[0].insert(round(randomIndex*len(nextstates[0])),previous["state"])
                    actions[0].insert(round(randomIndex*len(nextstates[0])),previous["action"])
            p +=1
        # Tous les x parties, on train nos modeles
        if (episode+1)%trainRate == 0 and len(states[0]) > (maxSize-trainRate*21):
            # On affiche des informations
            # On lance la fonction qui train automatiquement
            print("\n\n\nEpisode n°", episode+1)
            print("\n\nAgent1:")
            print("Length data: ",len(states[0])," rows\n")
            p1.update(states[p1.session],actions[p1.session],rewards[p1.session],nextstates[p1.session])

            # On affiche les stats
            p1.display_stats(rewards[0])

            # On fait une sauvegarde des modèles
            p1.model_save("against-humain/")


        # Si plus de 10000 éléments dans le tableau, on retire le dernier 
        if len(states[0]) > maxSize:
            states[0] = states[0][1:maxSize]
            nextstates[0] = nextstates[0][1:maxSize]
            actions[0] = actions[0][1:maxSize]
            rewards[0] = rewards[0][1:maxSize]
    
        # Débuter les entraînements à partir du moment ou l'ensemble est plein
        if(len(states[0]) > (maxSize-trainRate*21)):
            episode += 1

        counter += 1
