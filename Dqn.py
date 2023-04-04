import random
import numpy as np
import tensorflow as tf
import datetime

class DQNAgent:
    def __init__(self, game,session,loadModel=False, epsilon=0.9, learning_rate=0.01, discount_factor=0.95):
        """
        Dans le constructeur on retrouve le paramètre game qui contien le jeu, session, pour attribuer un numéro unique, un booléen 
        loadmodel pour choisi s'il faut ou non charger un modèle existant le facteur epsilon (ou exploration), 
        learning_rate (taux d'apprentissage) et le discount_factor pour privilégier plus ou moins les récompenses futures par rapport à celles
        obtensible immédiatement

        """
        self.game = game
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.stats = {"games": 0, "wins": 0, "egalites": 0, "meanReward": 0}
        self.session = session
        # On définit la forme de la grille de jeu
        self.input_shape = (self.game.height, self.game.width)
        # On définit le nombre d'actions possible (ici, le nombre de colonnes)
        self.num_actions = self.game.width
        # On construit le modèle
        self.build_model(loadModel)
        

    def build_model(self,load):
        """
        Cette fonction permet de construire le modèle de l'agent DQN. Il est créé en utilisant la classe Sequential de keras 
        qui permet de créer un modèle en enchainant des couches les unes après les autres. Ici:
        - une input étant un vecteur de n lines où n est le nombre de case du jeu (L*l du jeu)
        - 2 couches cachées de 64 neuronnes avec comme fonction d'activation relu
        - Et en sortie un vecteur de n lignes où n représente le nombre d'action possible (donc de colonnes) en activation linéaire 
            et non softmax car ce n'est pas une probabilité, la qualité d'une action ne dépend pas de la qualité des autres

        """
        if load:
            self.model = tf.keras.models.load_model(load)
        else:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=self.input_shape),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(self.num_actions, activation='linear')
            ])
        # On compile le modèle en spécifiant l'optimizer (Adam) et la fonction de perte (Mean Squared Error)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
            loss='mse',
            metrics=['accuracy']
        )
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def model_save(self, path):
        self.model.save(path)

    def set_epsilon(self, eps):
        self.epsilon=eps

    def set_learning_rate(self, eps):
        self.learning_rate=eps

    def display_stats(self, rewards=None):
        accuracy = self.stats["wins"]/self.stats["games"]
        print("\n \nStats de l'Agent ", self.session, 
        "\nepsilon: ", self.epsilon,
        "\nlearning rate: ", self.learning_rate,
        "\nNb games: ",self.stats["games"], 
        "\nNb wins: ",self.stats["wins"],
        "\nNb egalites: ",self.stats["egalites"],
        "\nNb loose: ", (self.stats["games"]-self.stats["wins"]-self.stats["egalites"]),
        "\nAccuracy: ",accuracy)
        if rewards:
            print("Reward per game avg: ", np.array(rewards).mean())

    def get_action(self, state):
        """
        Cette fonction permet de choisir une action pour l'agent DQN. Elle utilise un facteur d'exploration (epsilon) pour décider
        si l'agent doit explorer (choisir une action aléatoirement) ou utiliser son modèle pour choisir la meilleure action. 
        Cela permet de ne pas restreindre l'agent à sa connaissance qui au début de l'apprentissage est presque nulle
        """
        if random.random() < self.epsilon:
            # Choisir une action aléatoirement
            # print("Random action:")
            return (True, random.choice(self.game.legal_actions))
        else:
            # Utiliser le modèle pour prédire les valeurs Q pour chaque action
            # print(state)
            q_values = self.model.predict(state, verbose=0)[0]
            # print(q_values)
            bestAction = np.argmax(q_values)
            if bestAction not in self.game.legal_actions:
                return (False, bestAction)
            else:
                return (True, bestAction)
            # print("Action choisi", choosenAction)
            # choosenAction = np.argmax([q_values[i] if i in legal_actions else -float('inf') for i in range(self.num_actions)])
            # Retourne l'action qui a la valeur Q la plus élevée (parmi les actions légales)
            



    def update(self, state, actions, rewards, next_states, epochs=1,verbose=2):
        """
        Cette fonction permet de mettre à jour les valeurs Q de l'agent DQN en utilisant l'algorithme de Q-Learning.
        Elle prend en entrée l'état actuel, l'action choisie, la récompense obtenue, l'état suivant (après coup de l'adversaire) 
        et les actions autorisées dans cet état suivant.
        """
        # On utilise le modèle pour prédire les valeurs Q pour l'état actuel

        # On transforme les tableaux/liste en matrices
        npStates = np.array(state)
        npNextStates = np.array(next_states)
        npRewards = np.array(rewards)
        npActions = np.array(actions)

        size = len(npStates)
        npArangeSize = np.arange(size) #[0,1,2,3,..,n]

        # On met à jour la valeur Q uniquement pour l'action choisie
        # Choisi la prévision du prochain état avec 
        # SI valeur pas légale, -1 et on redemande de jouer 
        

        # Création des Q targets:

        # Calcul de l'eperence de la récompense maximale au prochain tour en prédisant avec l' état suivant
        next_q_values = self.model.predict(npNextStates,verbose=0)
        
        # On calcule les valeures que l'on aurait du avoir (estimation):
        # On récupère une matrice contenant les valeurs maximales prédites pour chaque lignes puis on applique la formule des Q values:
        # On multiple par le discount_factor (gamma) et auquel on ajoute les npRewards
        q_target_values = next_q_values[npArangeSize, np.argmax(next_q_values,1)] * self.discount_factor + npRewards
        
        # On récupère les Q valeurs pour l'état avant action
        q_target = self.model.predict(npStates, verbose=0)

        # On remplace uniquement les valeurs des actions CHOISI par les q_target
        q_target[npArangeSize, npActions] = q_target_values


        # On utilise cette nouvelle valeur pour entraîner le modèle q_target est la cible tandis que les q_values seront automatiquement
        # prédites à partir des npStates 
        # L'erreur (ici caclulcé avec la méthode mean squared error MSE) et le gradient se calcule aussi automatiquement
        #Le modèle va s'entraîner sur toute la longueur size des tableaux et va répéter l'opération epochs fois
        self.model.fit(npStates, q_target, epochs=epochs, verbose=verbose)
        
        # callbacks=[self.tensorboard_callback]
        # Permet une analyse graphique mais complexe à utiliser


        
        # A titre de débogage:
        # for i in range(size):
        #     print("States: ", npStates[i])
        #     print("\nNext States: ", npNextStates[i])
        #     print("\nAction: ", npActions[i])
        #     print("\nReward: ", npRewards[i])
        #     print("\nQ_value: ", tempValue[i])
        #     print("\nNext_Q_value: ", next_q_values[i])
        #     print("\nQ_value_target: ", q_target[i])
