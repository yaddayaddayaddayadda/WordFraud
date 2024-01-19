import numpy as np
import re
import copy


with open('SAOL_AUGMENTED.txt', 'r', encoding='utf8') as file:
    text = file.readlines()
    text = [re.sub(r'\n', '', word) for word in text]
    text = [re.sub(r'é', 'e', word).upper() for word in text]
    text = [word for word in text if "-" not in word]
    WORDLIST = set(text)

# CONSTANTS
HEIGHT, WIDTH = 15, 15
LETTER_TO_SCORE = {
    "A":1,
    "B":3,
    "C":8,
    "D":1,
    "E":1,
    "F":3,
    "G":2,
    "H":3,
    "I":1,
    "J":7,
    "K":3,
    "L":2,
    "M":3,
    "N":1,
    "O":2,
    "P":4,
    "R":1,
    "S":1,
    "T":1,
    "U":4,
    "V":3,
    "X":8,
    "Y":7,
    "Z":8,
    "1":4,
    "2":4,
    "3":4
}
class Engine:
    def __init__(self):
        pass
    
    def calculate_score(self, board, added_squares, direction):
        '''
        board is a 15 x 15 array with elements S where S is of the class Square
        S can also be a multiplier with the following meaning:
        DB - dubbel bokstav
        DO - dubbel ord
        TB - trippel boksta
        TO - trippel ord
        # - svart ruta

        added_squares is a list of tuples (S, (x,y)) with square S and position x,y, and direction is
        either equal to 'HOR' or 'VER' to indicate direction the letters are going in
        '''
        score = 0
        if len(added_squares) == 7:
            score += 40
        if direction == "HOR":
            rowScore = 0
            rowMultiplier = 1
            for square, (x,y) in added_squares:
                letter = square.letter
                isEligible = square.isEligible
                colScore = 0
                #calculate DOWN
                i=1
                while y+i<HEIGHT and board[y+i, x].letter in LETTER_TO_SCORE:
                    if board[y+i,x].isEligible:
                        colScore += LETTER_TO_SCORE[board[y+i,x].letter]
                    i+=1
                #calculate UP
                i=1
                while y-i>=0 and board[y-i,x].letter in LETTER_TO_SCORE:
                    if board[y-i,x].isEligible:
                        colScore += LETTER_TO_SCORE[board[y-i, x].letter]
                    i+=1
                hasNeighbor = self.hasNeighbor(board, (x,y), "VER")
                if board[y,x].letter == "DB":
                    if isEligible and hasNeighbor:
                        colScore += 2*LETTER_TO_SCORE[letter]
                elif board[y,x].letter == "TB":
                    if isEligible and hasNeighbor:
                        colScore += 3*LETTER_TO_SCORE[letter]
                elif board[y,x].letter == "DO":
                    if isEligible and hasNeighbor:
                        colScore += LETTER_TO_SCORE[letter]
                    colScore *= 2
                elif board[y,x].letter == "TO":
                    if isEligible and hasNeighbor:
                        colScore += LETTER_TO_SCORE[letter]
                    colScore *= 3
                else:
                    if isEligible and hasNeighbor:
                        colScore += LETTER_TO_SCORE[letter]
                score += colScore
            # calculate LEFT tail
            _, (x, y)  = added_squares[0]
            xes = {square[1][0] : square[0] for square in added_squares}
            x -= 1
            while x >= 0 and board[y, x].letter in LETTER_TO_SCORE:
                x -= 1
            x += 1
            while x < WIDTH:
                if board[y,x].letter in LETTER_TO_SCORE:
                    if board[y,x].isEligible:
                        rowScore += LETTER_TO_SCORE[board[y,x].letter]
                elif x in xes:
                    letter, isEligible = xes[x].letter, xes[x].isEligible
                    hasNeighbor = len(xes)>1 or self.hasNeighbor(board, (x,y), "HOR")
                    if board[y,x].letter == "DB":
                        if isEligible and hasNeighbor:
                            rowScore += 2*LETTER_TO_SCORE[letter]
                    elif board[y,x].letter == "TB" and isEligible:
                        if isEligible and hasNeighbor:
                            rowScore += 3*LETTER_TO_SCORE[letter]
                    elif board[y,x].letter == "DO":
                        if isEligible and hasNeighbor:
                            rowScore += LETTER_TO_SCORE[letter]
                        rowMultiplier *= 2
                    elif board[y,x].letter == "TO":
                        if isEligible and hasNeighbor:
                            rowScore += LETTER_TO_SCORE[letter]
                        rowMultiplier *= 3
                    else:
                        if isEligible and hasNeighbor:
                            rowScore += LETTER_TO_SCORE[letter]
                else:
                    break
                x += 1
            #print(f'Score = {score}, rowScore = {rowScore}, rowMultiplier = {rowMultiplier}')
            score += rowScore*rowMultiplier
        else:
            colScore = 0
            colMultiplier = 1
            for square, (x,y) in added_squares:
                letter = square.letter
                isEligible = square.isEligible
                rowScore = 0
                #calculate LEFT
                i = 1
                while x - i >= 0 and board[y, x - i].letter in LETTER_TO_SCORE:
                    if board[y,x-i].isEligible:
                        rowScore += LETTER_TO_SCORE[board[y, x-i].letter]
                    i+=1
                #calculate RIGHT
                i = 1
                while x + i < WIDTH and board[y, x + i].letter in LETTER_TO_SCORE:
                    if board[y,x+i].isEligible:
                        rowScore += LETTER_TO_SCORE[board[y, x + i].letter]
                    i+=1
                hasNeighbor = self.hasNeighbor(board, (x,y), "HOR")
                if board[y,x].letter == "DB":
                    if isEligible and hasNeighbor:
                        rowScore += 2*LETTER_TO_SCORE[letter]
                elif board[y,x].letter == "TB":
                    if isEligible and hasNeighbor:
                        rowScore += 3*LETTER_TO_SCORE[letter]
                elif board[y,x].letter == "DO":
                    if isEligible and hasNeighbor:
                        rowScore += LETTER_TO_SCORE[letter]
                    rowScore *= 2 
                elif board[y,x].letter == "TO":
                    if isEligible and hasNeighbor:
                        rowScore += LETTER_TO_SCORE[letter]
                    rowScore *= 3
                else:
                    if isEligible and hasNeighbor:
                        rowScore += LETTER_TO_SCORE[letter]
                score += rowScore
            # calculate UP tail
            _, (x, y) = added_squares[0]
            ys = {square[1][1] : square[0] for square in added_squares}
            y -= 1
            while y >= 0 and board[y, x].letter in LETTER_TO_SCORE:
                y-=1
            y += 1
            while y < HEIGHT:
                if board[y,x].letter in LETTER_TO_SCORE:
                    if board[y,x].isEligible:
                        colScore += LETTER_TO_SCORE[board[y,x].letter]
                elif y in ys:
                    letter, isEligible = ys[y].letter, ys[y].isEligible
                    hasNeighbor = len(ys)>1 or self.hasNeighbor(board, (x,y), "VER")
                    if board[y,x].letter == "DB":
                        if isEligible and hasNeighbor:
                            colScore += 2*LETTER_TO_SCORE[letter]
                    elif board[y,x].letter == "TB":
                        if isEligible and hasNeighbor:
                            colScore += 3*LETTER_TO_SCORE[letter]
                    elif board[y,x].letter == "DO":
                        if isEligible and hasNeighbor:
                            colScore += LETTER_TO_SCORE[letter]
                        colMultiplier *= 2
                    elif board[y,x].letter == "TO":
                        if isEligible and hasNeighbor:
                            colScore += LETTER_TO_SCORE[letter]
                        colMultiplier *= 3
                    else:
                        if isEligible and hasNeighbor:
                            colScore += LETTER_TO_SCORE[letter]
                else:
                    break
                y += 1
            
            #print(f'score = {score}, colScore = {colScore}, colMultiplier = {colMultiplier}')
            score += colScore*colMultiplier
        return score

    def calculate_added_squares(self, board, squares, start_pos, direction):
        added_squares = []
        curX, curY = start_pos
        if direction == "HOR":
            while 0 <= curX < WIDTH and 0 <= curY < HEIGHT and squares:
                if board[curY, curX].letter in LETTER_TO_SCORE:
                    curX += 1
                else:
                    square = squares.pop(0)
                    added_squares.append((square, (curX, curY)))
                    curX += 1
        else:
            while 0 <= curX < WIDTH and 0 <= curY < HEIGHT and squares:
                if board[curY, curX].letter in LETTER_TO_SCORE:
                    curY += 1
                else:
                    square = squares.pop(0)
                    added_squares.append((square, (curX, curY)))
                    curY += 1
        return added_squares

    def is_legal(self, board, added_letters, direction):
        if not added_letters:
            return False
        letters = ''.join([letter for letter, _ in added_letters])
        if direction == "HOR":
            added_letters = sorted(added_letters, key = lambda x : x[1][0])
            _, (x,y) = added_letters[0]
            hasNeighbor = False
            for _, (x,y) in added_letters:
                if (x-1 >= 0 and board[y, x-1].letter in LETTER_TO_SCORE) or (x+1 < WIDTH and board[y, x+1].letter in LETTER_TO_SCORE) or (y-1 >= 0 and board[y-1, x].letter in LETTER_TO_SCORE) or (y+1 < HEIGHT and board[y+1, x].letter in LETTER_TO_SCORE):
                    hasNeighbor = True
                    break
            if not hasNeighbor:
                return False
            for letter, (x,y) in added_letters:
                columnWord = letter
                #calculate DOWN
                i=1
                while y+i<HEIGHT and board[y+i, x].letter in LETTER_TO_SCORE:
                    columnWord += board[y+i,x].letter
                    i += 1
                #calculate UP
                i=1
                while y-i>=0 and board[y-i,x].letter in LETTER_TO_SCORE:
                    columnWord = board[y-i, x].letter + columnWord
                    i+=1
                columnWord = re.sub("1", "Å", columnWord)
                columnWord = re.sub("2", "Ä", columnWord)
                columnWord = re.sub("3", "Ö", columnWord)
                if columnWord not in WORDLIST and columnWord != letter:
                    return False
            # step to the LEFT
            _, (x_left, y_left)  = added_letters[0]
            xes = {letter[1][0] : letter[0] for letter in added_letters}
            x_left -= 1
            while x_left  >= 0 and board[y_left, x_left ].letter in LETTER_TO_SCORE:
                x_left -= 1
            x_left += 1
            rowWord = ''
            while x_left < WIDTH:
                if x_left in xes:
                    rowWord += xes[x_left]
                elif board[y_left, x_left].letter in LETTER_TO_SCORE:
                    rowWord += board[y_left, x_left].letter
                else:
                    break
                x_left += 1
            rowWord = re.sub("1", "Å", rowWord)
            rowWord = re.sub("2", "Ä", rowWord)
            rowWord = re.sub("3", "Ö", rowWord)
            if rowWord not in WORDLIST:
                return False
        else:
            added_letters = sorted(added_letters, key = lambda x : x[1][1])
            _, (x,y) = added_letters[0]
            hasNeighbor = False
            for _, (x,y) in added_letters:
                if (x-1 >= 0 and board[y, x-1].letter in LETTER_TO_SCORE) or (x+1 < WIDTH and board[y, x+1].letter in LETTER_TO_SCORE) or (y-1 >= 0 and board[y-1, x].letter in LETTER_TO_SCORE) or (y+1 < HEIGHT and board[y+1, x].letter in LETTER_TO_SCORE):
                    hasNeighbor = True
                    break
            if not hasNeighbor:
                return False
            for letter, (x,y) in added_letters:
                rowWord = letter
                #calculate LEFT
                i = 1
                while x - i >= 0 and board[y, x - i].letter in LETTER_TO_SCORE:
                    rowWord = board[y, x-i].letter + rowWord
                    i+=1
                #calculate RIGHT
                i = 1
                while x + i < WIDTH and board[y, x + i].letter in LETTER_TO_SCORE:
                    rowWord += board[y, x + i].letter
                    i+=1
                rowWord = re.sub("1", "Å", rowWord)
                rowWord = re.sub("2", "Ä", rowWord)
                rowWord = re.sub("3", "Ö", rowWord)
                if rowWord not in WORDLIST and rowWord != letter:
                    return False
            # step UP
            _, (x_up, y_up) = added_letters[0]
            ys = {letter[1][1] : letter[0] for letter in added_letters}
            y_up -= 1
            while y_up >= 0 and board[y_up, x_up].letter in LETTER_TO_SCORE:
                y_up -= 1
            y_up += 1
            colWord = ''
            while y_up < HEIGHT:
                if y_up in ys:
                    colWord += ys[y_up]
                elif board[y_up, x_up].letter in LETTER_TO_SCORE:
                    colWord += board[y_up, x_up].letter
                else:
                    break
                y_up += 1
            colWord = re.sub("1", "Å", colWord)
            colWord = re.sub("2", "Ä", colWord)
            colWord = re.sub("3", "Ö", colWord)
            if colWord not in WORDLIST:
                return False
        return True
        
    def hasNeighbor(self, board, pos, direction):
        x,y = pos
        if direction == "HOR":
            return (x>0 and board[y, x-1].letter in LETTER_TO_SCORE) or (x<WIDTH-1 and board[y, x+1].letter in LETTER_TO_SCORE)
        else:
            return (y>0 and board[y-1, x].letter in LETTER_TO_SCORE) or (y<HEIGHT-1 and board[y+1, x].letter in LETTER_TO_SCORE)