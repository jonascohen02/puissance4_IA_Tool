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
        Cette fonction permet de construire le modèle de l'agent DQN reposant sur le deep Q learning. Il est créé en utilisant la classe Sequential de keras du module tensorflow
        qui permet de créer un modèle en enchainant des couches les unes après les autres. Ici:
        - une input étant un vecteur de n lines où n est le nombre de case du jeu (L*l du jeu)
        - 2 couches cachées de 64 neuronnes avec comme fonction d'activation relu
        - Et en sortie un vecteur de n lignes où n représente le nombre d'action possible (donc de colonnes) en activation linéaire 
            et non softmax car ce n'est pas une probabilité, la pertinence d'une action ne dépend pas de la pertinence des autres, 2 actions peuvent être très pertinentes si on
            peut gagenr de 2 manières par exemple.

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

    def set_gamma(self, eps):
        self.discount_factor=eps

    def display_stats(self, rewards=None):
        """
        Affiche des informations pour suivre la progression de notre IA et connaître son évolution
        """
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
            return (True, random.choice(self.game.legal_actions))
        else:
            # Utiliser le modèle pour prédire les valeurs Q pour chaque action
            q_values = self.model.predict(state, verbose=0)[0]
            # On séléctionne la valeur la plus élevé
            bestAction = np.argmax(q_values)
            # Si l'action choisi est interdite, on retourne False pour indiquer une mauvaise action ainsi que cette dernière
            if bestAction not in self.game.legal_actions:
                return (False, bestAction)
            else:
                return (True, bestAction)            



    def update(self, state, actions, rewards, next_states, epochs=1,verbose=2):
        """
        Cette fonction permet de mettre à jour les valeurs Q de l'agent DQN en utilisant l'algorithme de Q-Learning 
        reposant sur l'équations de Bellman (Semblable à une suite défini par réccurence avec 2 variables):
        "La Q value à l'état S et avec l'action A à l'instant t est égale à la récompense à cet instant + discount_factor * la q valeur maximale à t+1"
        Elle prend donc en entrée l'état initiale, l'action choisie, la récompense obtenue, l'état suivant (après coup de l'adversaire).
        """
        # On utilise le modèle pour prédire les valeurs Q pour l'état actuel

        # On transforme les tableaux/liste en matrices pour accélérer les calculs (plus de 10000 lignes à chaque calcul)
        npStates = np.array(state)
        npNextStates = np.array(next_states)
        npRewards = np.array(rewards)
        npActions = np.array(actions)

        # Longueur de l'échantillon
        size = len(npStates)

        # Utile pour les opérations matricielles
        npArangeSize = np.arange(size) #[0,1,2,3,..,n]

        # On met à jour la valeur Q uniquement pour l'action choisie
        # Choisi la prévision du prochain état avec 
        # SI valeur pas légale, -1 et on redemande de jouer 
        

        # Création des Q targets:

        # Calcul de l'eperence de la récompense maximale au prochain tour en prédisant avec l'état suivant
        next_q_values = self.model.predict(npNextStates,verbose=0)
        
        # On calcule les valeures que l'on aurait du avoir (estimation) pour la Q value:
        # On récupère une matrice contenant les valeurs maximales prédites pour chaque lignes puis on applique l'équation de Bellman:
        # On multiple par le discount_factor (gamma) et auquel on ajoute les npRewards
        # On obtient une matrice contenant toutes les q_values que le modèle aurait du prédire
        q_target_values = next_q_values[npArangeSize, np.argmax(next_q_values,1)] * self.discount_factor + npRewards
        
        # On récupère une matrice contenant les Q values préditess par le modèle pour l'état S et on va modifier l'action choisi
        # par la q_target_values pour chaque ligne de la matrice dans l'instruction d'après
        q_target = self.model.predict(npStates, verbose=0)

        # Ici, on remplace donc uniquement les valeurs de l'action CHOISI pour chaque ligne par la q_target_value
        q_target[npArangeSize, npActions] = q_target_values


        # On utilise cette nouvelle valeur pour entraîner le modèle q_target est la cible tandis que les q_values seront automatiquement
        # prédites à partir des npStates 
        # L'erreur (ici caclulcé avec la méthode mean squared error MSE qui représente l'écart entre les q_target et les q prédites) et le gradient se calcule aussi automatiquement
        # Le modèle va s'entraîner sur toute la longueur size des matrices et va répéter l'opération epochs fois
        # Pour un échantillon de 10 000 états/action/rewards avec une époques de 100, on aura donc 1 Milion d'itération (sans matrice),
        # simplifié à 100 calculs matriciels de 10 000 lignes chacun
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
