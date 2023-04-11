import numpy as np

class Game:
    def __init__(self, height, width):
        self.grid = []
        self.height = height
        self.width = width
        self.reset()

    def reset(self):
        """
        Vide la grille
        """
        self.grid = []
        # On ajoute dans une liste pour chaque ligne une liste de n 0 pour symboliser une case vide
        # Attention, les index des lignes de la grille sont inversés: 

        # Exemple pour une grille de 6,7
    # grid  = [
    #     [0,0,0,0,0,0,0],       Index 0
    #     [0,0,0,0,0,0,0],       Index 1
    #     [0,0,0,0,0,0,0],       Index 2
    #     [0,0,0,0,0,0,0],       Index 3
    #     [0,0,0,0,0,0,0],       Index 4
    #     [0,0,0,0,0,0,0]        Index 5
    # ]

        # La ligne du bas correspond donc au dernier index
        for i in range(self.height):
            self.grid.append([0] * self.width)
        self.player = 1
        self.legal_actions = list(range(self.width))
        self.winner = None


    def convert_numbers_to_char(self, number):
        """
        Cette fonction convertit chaque nombre contenu dans la grille en caractère plus compréhensible pour l'humain
        """
        if number == 1:
            return "X"
        elif number == 2:
            return "O"
        else:
            return "."
    # Fonction pour afficher la grille de jeu

    def display_grid(self):
        """
        Permet d'afficher la matrice sous forme de grille
        """
        # Met correctement le bon espacement en fonction du nombre de chiffres dans le nombre de colonnes Si il y a + 10 colonnes,
        # il faut 2 espaces pour écrire le 1 et le 0
        lenW = len(str(self.width))
        # Affiche les indicateurs de colonnes (1,2,3,.., n)
        print(" ".join(["{:^{}}".format(i+1,lenW) for i in range(self.width)]))
        for row in self.grid:
            displayRow = map(self.convert_numbers_to_char, row)
            print((" "*lenW).join(displayRow))
        # print("Player {}".format(self.player))


    def add_piece(self, column):
        """
        Ajoute un pion dnas la colonne choisie
        """
        # Pour chaque on ligne de la colonne, en partant du bas (la grille est inversé car la première ligne ajoutée se retrouve en haut
        # donc du plus grand vers le plus petit, voir commentaire de la méthode reset classe Game), on vérifie que la case est vide (=0),si c'est le cas on lui affecte la valeur du joueur
        for row in range(self.height-1, -1, -1):
            if self.grid[row][column] == 0:
                self.grid[row][column] = self.player
                playerPiece = (row, column) 

                # Si la ligne la plus haute vient d'être occupée (grille inversé donc la première), on ne peux plus placer de pion
                if self.grid[0][column] == self.player:
                # On supprime la possibilité de choisir une colonne pleine. On utilise pas remove car sinon il assigne 
                # une donnée à 2 variables:  legal_actions et aussi states d'entrainement (il fait une reference et non une copie)
                # -10
                    self.legal_actions = [i for i in self.legal_actions if i != column]
                break
        self.winner = self.check_win()
        reward = self.get_reward(*playerPiece)            
        self.player = 2 if self.player == 1 else 1
        return (self.winner, reward)

    def get_reward(self, row, col):
        """
        Détermine la récompense à attribuer à l'IA
        """
        reward = 0
        # Conversion de la liste de liste en matrice pour accélérer et faciliter les calculs
        npGrid = np.array(self.grid)
        isBlocking = ['verticalBottom', 'horizontalRight', 'horizontalLeft', 'horizontalCenteredRight', 'horizontalCenteredLeft', 'diagonalBottomRight', 'diagonalBottomLeft', 'diagonalTopRight', 'diagonalTopLeft', 'diagonalCenteredBottomRight', 'diagonalCenteredBottomLeft', 'diagonalCenteredTopLeft', 'diagonalCenteredTopRight']

        # Liste de 3 valeurs de l'adversaire à comparer
        opponent = 2 if self.player == 1 else 1
        validateArray = [opponent]*3
        
        # Par colonne
        # On vérifie uniquement les 3 cellules en dessous de celle qu'on vient de poser
        isBlocking[0] = np.array_equal(npGrid[row+1:row+4,col], validateArray)

        # Par ligne
        # 3 pions adverses à droite de la pièce ?
        isBlocking[1] = np.array_equal(npGrid[row,col+1:col+4], validateArray)
        # 3 pions adverses à gauche de la pièce ?
        isBlocking[2] = np.array_equal(npGrid[row,col-3:col], validateArray)
        # 1 à gauche et 2 à droite ?
        isBlocking[3] = np.array_equal(np.concatenate((npGrid[row,col-1:col], npGrid[row,col+1:col+3])), validateArray)
        # 1 à droite et 2 à gauche ?
        isBlocking[4] = np.array_equal(np.concatenate((npGrid[row,col-2:col], npGrid[row,col+1:col+2])), validateArray)
        
        # En diagonales
        # Il faut vérifier pièce par pièce chaque diagonale possible pour éviter des erreurs comme on travaille avec des matrices
        # (pas de possibilité de séléctionner une diagonale si elle comprend des index non définies) 
        # diagonalle est du haut à gauche vers le bas à droite  contrairement à sa diagonale inversé par rapportà l'axe x: du coin en haut à droite vers bas à gauche

        # 3 pions diagonale vers le bas à droite ?
        isBlocking[5] = np.array_equal(np.concatenate((npGrid[row+1:row+2,col+1:col+2].flatten(), npGrid[row+2:row+3,col+2:col+3].flatten(), npGrid[row+3:row+4,col+3:col+4].flatten())), validateArray)
        # 3 pions diagonale vers le bas à gauche ? (diagonale inversé symétrique sur l'axe x)
        isBlocking[6] = np.array_equal(np.concatenate((npGrid[row+1:row+2,col-1:col].flatten(), npGrid[row+2:row+3,col-2:col-1].flatten(), npGrid[row+3:row+4,col-3:col-2].flatten())), validateArray)
        # 3 pions diagonale vers le haut à droite ?
        isBlocking[7] = np.array_equal(np.concatenate((npGrid[row-1:row,col+1:col+2].flatten(), npGrid[row-2:row-1,col+2:col+3].flatten(), npGrid[row-3:row-2,col+3:col+4].flatten())), validateArray)
        # 3 pions diagonale vers le haut à gauche ? (diagonale inversé symétrique sur l'axe x)
        isBlocking[8] = np.array_equal(np.concatenate((npGrid[row-1:row,col-1:col].flatten(), npGrid[row-2:row-1,col-2:col-1].flatten(), npGrid[row-3:row-2,col-3:col-2].flatten())), validateArray)  
        # 2 pions diagonale vers le bas à droite et 1 en haut à gauche ?
        isBlocking[9] = np.array_equal(np.concatenate((npGrid[row-1:row,col-1:col].flatten(), npGrid[row+1:row+2,col+1:col+2].flatten(), npGrid[row+2:row+3,col+2:col+3].flatten())), validateArray)
        # 2 pions diagonale vers le bas à gauche et 1 en haut à droite ? (diagonale inversé symétrique sur l'axe x)
        isBlocking[10] = np.array_equal(np.concatenate((npGrid[row-1:row,col+1:col+2].flatten(), npGrid[row+1:row+2,col-1:col].flatten(), npGrid[row+2:row+3,col-2:col-1].flatten())), validateArray)
        # 2 pions diagonale vers le haut à gauche et 1 en bas à droite ?
        isBlocking[11] = np.array_equal(np.concatenate((npGrid[row+1:row+2,col+1:col+2].flatten(), npGrid[row-1:row,col-1:col].flatten(), npGrid[row-2:row-1,col-2:col-1].flatten())), validateArray)
        # 2 pions diagonale vers le haut à droite et 1 en bas à gauche ? (diagonale inversé symétrique sur l'axe x)
        isBlocking[12] = np.array_equal(np.concatenate((npGrid[row+1:row+2,col-1:col].flatten(), npGrid[row-1:row,col+1:col+2].flatten(), npGrid[row-2:row-1,col+2:col+3].flatten())), validateArray)

        # S'il y a au moins 1 blocage, on récompense de 5
        if any(isBlocking):
            reward = 5
        
        return reward



    def check_win(self):
        """
        On vérifie si il y a un vainqueur ou uné égalité
        """
        # Pour chaque ligne, on prend 4 points à partir de la première colonne, et on voit s'ils sont égaux au joueur qui vient de jouer
        # On s'arrète à la n-3e colonne car il faut minimum 4 cellules pour gagner
        for row in self.grid:
            for i in range(self.width-3):
                if row[i:i+4] == [self.player]*4:
                    return self.player
                

        # Idem que pour la ligne mais vérticalement:
        # Pour chaque colonnes, on prend 4 points à partir de la première ligne (le fait que la grille soit inversé,cf:commentaire reset classe Game, n'affecte pas la vérification), et on voit s'ils sont égaux au joueur qui vient de jouer
        # On s'arrète à la n-3e ligne (3 lignes avant le bas) car il faut minimum 4 cellules pour gagner
        for col in range(self.width):
            for i in range(self.height-3):
                if self.grid[i][col] == self.player and self.grid[i+1][col] == self.player and self.grid[i+2][col] == self.player and self.grid[i+3][col] == self.player:
                    return self.player
                

        # Pour chaque cellule, on vérifie que les cellules (sauf celles de la zone morte col-3 et l-3 représenté par M ci après, car il faut 4 cellules au minimum pour gagner) 
        # de la diagonale de en bas à droite, vers en bas à droite par raport à la cellule que l'on étudie, sont égaux au joueur qui vient de jouer.
        # [0,0,0,0,M,M,M],       
        # [0,X,0,0,M,M,M],      
        # [0,0,V,0,M,M,M],      
        # [M,M,M,V,M,M,M],      
        # [M,M,M,M,V,M,M],      
        # [M,M,M,M,M,M,M]  
        # On vérifie dans cette grille pour chaque cellule X contenu dans la zone non morte séléctionné que X=V=V=V
        for row in range(self.height-3):
            for col in range(self.width-3):
                if self.grid[row][col] == self.player and self.grid[row+1][col+1] == self.player and self.grid[row+2][col+2] == self.player and self.grid[row+3][col+3] == self.player:
                    return self.player
                
        # Pour chaque cellule, on vérifie que les cellules (sauf celles de la zone morte col-3 et l-3 représenté par M ci après, car il faut 4 cellules au minimum pour gagner) 
        # de la diagonale de en bas à gauche, vers en haut à droite par raport à la cellule que l'on étudie, sont égaux au joueur qui vient de jouer.
        # [0,0,0,0,M,M,M],       
        # [0,X,0,0,V,M,M],      
        # [0,0,0,V,M,M,M],      
        # [M,M,V,M,M,M,M],      
        # [M,V,M,M,M,M,M],      
        # [M,M,M,M,M,M,M]  
        # On vérifie dans cette grille pour chaque cellule X contenu dans la zone non morte séléctionné que V=V=V=V, X sert ici de référent mais n'est pas vérifié
        for row in range(self.height-3):
            for col in range(self.width-3):
                 if self.grid[row+3][col] == self.player and self.grid[row+2][col+1] == self.player and self.grid[row+1][col+2] == self.player and self.grid[row][col+3] == self.player:
                    return self.player
                 
        # Si toutes les cellules sont pleines, on renvoie "egal"
        for row in self.grid:
            for col in row:
                if col == 0:
                    return None
        return "egal"