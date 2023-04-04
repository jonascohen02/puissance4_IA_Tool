# puissance4_IA_Tool

Le fichier Env contient l'environnement du puissance 4
Le fichier Dqn contient l'agent Deep Q Network (Intelligence artificielle reposant sur l'approximation du Q learning et des équations de Bellman par un autre modèle de deep learning)
Le fichier functions, contient les fonctions principales pour jouer des parties, et entraîner le modèle

Le fichier main est le script principale à lancer pour entraîner le modèle et lancer les parties:

1) Il suffit d'y créer l'environnement:
game = Game(hauteur, largeur), pour créer une grille de 6*7 par ex: game = Game(6, 7)

2) Executer les fonctions pour créer l'agent:
agent = DQNAgent(game,session,loadModel=False, epsilon=0.9, learning_rate=0.01, discount_factor=0.95)
La classe DQNAgent prend ainsi en paramètre:
- l'environnement (game)
- le numéro de la session (pour les différencier)
Facultatif:
- si besoin, on peut charger un modèle préentraîné en spécifiant le chemin d'accès
- le facteur chance
- le facteur de taux d'apprentissage
- le facteur gamma (privilégier les actions futures lointaines à celles proches)

3) Puis de lancer l'entraînement 
play_episode(p1,p2,numberEpisodes, trainRate, game, maxSize=10000) qui prend comme paramètres:
- Agent 1
- Agent 2 (on fait jouer en général 2 IA différente l'une contre l'autre)
- le nombre d'épisodes à faire (combien de parties)
- Toutes les combiens de parties, l'entraînement doit-il se faire (en général, environ 20) en reprenant toutes les aprties effectués limités par la taille de l'échantillon 
- un environnement (game)
- la longueur maximale de l'échantillon d'états/actions à utiliser (par défaut de 10 000, dépendant de la mémoire vive de votre ordinateur)

L'entraînement démarre quand la taille de l'échantillon (length data) est environ égale à la longueur maximum

La fonction va faire des sauvegardes des 2 modèles automatiquement toutes les trainRate parties dans un dossier ia1/ et ia2/
On peut ensuite sauvegarder les modèles à d'autres emplacement


on peut aussi réduire le facteur chance/epsilon, la vitesse d'apprentissage/learning_rate et le gamma/discount_factor au cours du temps avec les instruction: 
agent.set_epsilon('Nouvelle valeur')
agent.set_learning_rate('Nouvelle valeur')
agent.set_gamma('Nouvelle valeur')

On peut afficher les stats de l'agent avec:
agent.display_stats()

5) On sauvegarde les agents (si on souhaite les réutiliser)
agent1.model_save("chemin/")
agent2.model_save("chemin2/")

4) Jouer contre l'agent entraîné: 
play_against(p1, game, train=False, numberEpisodes=1, trainRate=None, maxSize=10000) prenant en paramètre:
- p1: l'agent à affronter
- game: l'environnement dans lequel jouer
Facultatif:
- numberEpisodes: le nombre de parties à jouer
- train: s'il faut ou non entraîner le modèle pendant qu'il joue contre vous
Obligatoire si train=True:
- trainRate: Toutes les combiens de parties, l'entraînement doit-il se faire (en général, environ 20) en reprenant toutes les parties effectués dans la limite de la taille maxSize
- maxSize: la longueur maximale de l'échantillon d'états/actions à utiliser (par défaut de 10 000, dépendant de la mémoire vive de votre ordinateur)
Si train=True, 
La fonction va faire des sauvegardes des 2 modèles automatiquement toutes les trainRate parties dans un dossier against-humain/
On peut ensuite sauvegarder les modèles à d'autres emplacement