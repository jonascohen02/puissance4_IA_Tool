import numpy as np

class Game:
    def __init__(self, height, width):
        self.grid = []
        self.height = height
        self.width = width
        self.reset()

    def reset(self):
        self.grid = []
        for i in range(self.height):
            self.grid.append([0] * self.width)
        self.player = 1
        self.legal_actions = list(range(self.width))
        self.winner = None


    def convert_numbers_to_char(self, number):
        if number == 1:
            return "X"
        elif number == 2:
            return "O"
        else:
            return "."
    # Fonction pour afficher la grille de jeu

    def display_grid(self):
        # Met correctement le bon espacement en fonction du nombre de chiffres dans le nombre de colonnes
        lenW = len(str(self.width))
        print(" ".join(["{:^{}}".format(i+1,lenW) for i in range(self.width)]))
        for row in self.grid:
            displayRow = map(self.convert_numbers_to_char, row)
            print((" "*lenW).join(displayRow))



    def add_piece(self, column):
        for row in range(self.height-1, -1, -1):
            if self.grid[row][column] == 0:
                self.grid[row][column] = self.player
                playerPiece = (row, column) 

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

    # Accorder une récompense si il a bloqué l'adversaire
    def get_reward(self, row, col):
        reward = 0
        npGrid = np.array(self.grid)
        opponent = 2 if self.player == 1 else 1
        validateArray = [opponent]*3
        isBlocking = ['verticalBottom', 'horizontalRight', 'horizontalLeft', 'horizontalCenteredRight', 'horizontalCenteredLeft', 'diagonalBottomRight', 'diagonalBottomLeft', 'diagonalTopRight', 'diagonalTopLeft', 'diagonalCenteredBottomRight', 'diagonalCenteredBottomLeft', 'diagonalCenteredTopLeft', 'diagonalCenteredTopRight']
        
        # ALONG COLUMNS
        # Not other choices because piece just set
        isBlocking[0] = np.array_equal(npGrid[row+1:row+4,col], validateArray)

        # ALONG ROWS
        # Right to piece:
        # self.grid[row][col+1:col+4]
        isBlocking[1] = np.array_equal(npGrid[row,col+1:col+4], validateArray)
        # Left:
        # self.grid[row][col-3:col]
        isBlocking[2] = np.array_equal(npGrid[row,col-3:col], validateArray)
        # Centerd right
        # self.grid[row][col-1:col] + self.grid[row][col+1:col+3]
        isBlocking[3] = np.array_equal(np.concatenate((npGrid[row,col-1:col], npGrid[row,col+1:col+3])), validateArray)
        # Centered left
        # self.grid[row][col-2:col] + self.grid[row][col+1:col+2]
        isBlocking[4] = np.array_equal(np.concatenate((npGrid[row,col-2:col], npGrid[row,col+1:col+2])), validateArray)
        
        # ALONG DIAGONALS
        # Slicing to avoid getting exception
        # Struggle with diagonal because of the lot of combinaisons possible. Did this, piece by piece.
        # Reverse diagonal is from to right top corner to bottom left corner counter to "diagonal" which is from top left corner to bottom right corner

        # Bottom Right
        isBlocking[5] = np.array_equal(np.concatenate((npGrid[row+1:row+2,col+1:col+2].flatten(), npGrid[row+2:row+3,col+2:col+3].flatten(), npGrid[row+3:row+4,col+3:col+4].flatten())), validateArray)
        # Bottom Left (reverse diagonal) 
        isBlocking[6] = np.array_equal(np.concatenate((npGrid[row+1:row+2,col-1:col].flatten(), npGrid[row+2:row+3,col-2:col-1].flatten(), npGrid[row+3:row+4,col-3:col-2].flatten())), validateArray)
        # Top Right
        isBlocking[7] = np.array_equal(np.concatenate((npGrid[row-1:row,col+1:col+2].flatten(), npGrid[row-2:row-1,col+2:col+3].flatten(), npGrid[row-3:row-2,col+3:col+4].flatten())), validateArray)
        # Top Left (reverse diagonal)
        isBlocking[8] = np.array_equal(np.concatenate((npGrid[row-1:row,col-1:col].flatten(), npGrid[row-2:row-1,col-2:col-1].flatten(), npGrid[row-3:row-2,col-3:col-2].flatten())), validateArray)  
        # Centered Right Bottom oriented
        isBlocking[9] = np.array_equal(np.concatenate((npGrid[row-1:row,col-1:col].flatten(), npGrid[row+1:row+2,col+1:col+2].flatten(), npGrid[row+2:row+3,col+2:col+3].flatten())), validateArray)
        # Centered Left Bottom oriented (reverse diagonal)
        isBlocking[10] = np.array_equal(np.concatenate((npGrid[row-1:row,col+1:col+2].flatten(), npGrid[row+1:row+2,col-1:col].flatten(), npGrid[row+2:row+3,col-2:col-1].flatten())), validateArray)
        # Centered Left Top oriented
        isBlocking[11] = np.array_equal(np.concatenate((npGrid[row+1:row+2,col+1:col+2].flatten(), npGrid[row-1:row,col-1:col].flatten(), npGrid[row-2:row-1,col-2:col-1].flatten())), validateArray)
        # Centered Right Top oriented (reverse diagonal)
        isBlocking[12] = np.array_equal(np.concatenate((npGrid[row+1:row+2,col-1:col].flatten(), npGrid[row-1:row,col+1:col+2].flatten(), npGrid[row-2:row-1,col+2:col+3].flatten())), validateArray)

        if any(isBlocking):
            reward = 5
        
        return reward



    def check_win(self):
        for row in self.grid:
            for i in range(self.width-3):
                if row[i:i+4] == [self.player]*4:
                    return self.player
        for col in range(self.width):
            for i in range(self.height-3):
                if self.grid[i][col] == self.player and self.grid[i+1][col] == self.player and self.grid[i+2][col] == self.player and self.grid[i+3][col] == self.player:
                    return self.player
        for row in range(self.height-3):
            for col in range(self.width-3):
                if self.grid[row][col] == self.player and self.grid[row+1][col+1] == self.player and self.grid[row+2][col+2] == self.player and self.grid[row+3][col+3] == self.player:
                    return self.player
        for row in range(self.height-3):
            for col in range(self.width-3):
                 if self.grid[row+3][col] == self.player and self.grid[row+2][col+1] == self.player and self.grid[row+1][col+2] == self.player and self.grid[row][col+3] == self.player:
                    return self.player
        for row in self.grid:
            for col in row:
                if col == 0:
                    return None
        return "egal"