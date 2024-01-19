import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
import pickle
from skimage.segmentation import clear_border
import glob
from score_calculator import Engine
from itertools import permutations
import copy
import time
import heapq
from square import Square
import re
import argparse

ORG = (207, 233)
FONT = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 1.5
COLOR = (255, 255, 255)
THICKNESS = 2


ALPHABET = {
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
class Solver:
    def __init__(self, capacity):
        self.load_modules()
        self.capacity = capacity

    def determine_type(self, cell):
        R=np.median(cell[:,:,0])
        B=np.median(cell[:,:,1])
        G=np.median(cell[:,:,2])
        if R+G+B > 500:
            return "WHITE"
        elif B-R > 30 and B - G > 20:
            return "DB"
        elif R<100 and G<100 and B<100:
            return "#"
        elif G-B > 40 and B - R > 60:
            return "DO"
        elif R-G > 40 and B - G > 20:
            return "TB"
        elif G-B > 50 and abs(R-B) < 20:
            return "TO"
    def determine_type_playable_letter(self, cell):
        R=np.median(cell[:,:,0])
        B=np.median(cell[:,:,1])
        G=np.median(cell[:,:,2])
        if R<100 and G<100 and B<100:
            return False
        return True

    def load_modules(self):
        self.letterModel = load_model('letterclassifier')
        self.playableLetterModel = load_model('playableletterclassifier')
        self.scoreModel = load_model('scoreclassifier')
        with open('scoreTokenizer', 'rb+') as f:
            self.scoreLb=pickle.load(f)
        with open('letterTokenizer', 'rb+') as f:
            self.letterLb=pickle.load(f)
        with open('playableLetterTokenizer', 'rb+') as f:
            self.playableLetterLb=pickle.load(f)
    def load_image_from_file(self, file):
        self.image = cv2.imread(file)
    def load_image_from_np_arr(self, arr):
        self.image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def scale_image(self):
        if self.image.shape == (2532, 1170, 3):
            self.VERSION = "IPHONE13PRO"
        elif self.image.shape == (2556, 1179, 3):
            self.VERSION = "IPHONE14PRO"
        if self.VERSION == "IPHONE13PRO":
            self.image[140:270, 80:1040] = self.image[0,0]
        elif self.VERSION == "IPHONE14PRO":
            self.image[160:270, 80:1040] = self.image[0,0]
        if self.VERSION == "IPHONE13PRO":
            self.board = self.image[606:1776, :]
        elif self.VERSION == "IPHONE14PRO":
            self.board = self.image[624:1800, :]
        self.board = cv2.resize(self.board, (1170, 1170), interpolation=cv2.INTER_LINEAR)
        self.grayBoard = cv2.cvtColor(self.board, cv2.COLOR_BGR2GRAY)
        self.threshBoard = cv2.threshold(self.grayBoard, 40, 255, cv2.THRESH_BINARY)[1]
    def segment_image(self):
        stepX, stepY = self.threshBoard.shape[1]//15, self.threshBoard.shape[0]//15
        self.res = []
        for y in range(15):
            # initialize the current list of cell locations
            row = []
            for x in range(15):
                # compute the starting and ending (x, y)-coordinates of the
                # current cell
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1) * stepX
                endY = (y + 1) * stepY
                # add the (x, y)-coordinates to our cell locations list
                cell = self.board[startY:endY, startX:endX]
                cellType = self.determine_type(cell)
                #board is a 15 x 15 array with elements (L, S),
                if cellType == "WHITE":
                    whiteCell = self.threshBoard[startY:endY, startX:endX]
                    whiteCell = whiteCell.reshape(1,78,78,1)/255.0
                    letter=self.letterLb.inverse_transform(self.letterModel(whiteCell, training = False).numpy())
                    isScore=self.scoreLb.inverse_transform(self.scoreModel(whiteCell, training = False).numpy())
                    row.append((Square(letter[0][0], 1-isScore[0])))
                else:
                    row.append((Square(cellType, False)))
            self.res.append(row)

        if self.VERSION == "IPHONE13PRO":
            playable_letters = self.image[2122:2282,:]
        elif self.VERSION == "IPHONE14PRO":
            playable_letters = self.image[2146:2306,:]
        playable_letters = cv2.resize(playable_letters, (1170, 160), cv2.INTER_LINEAR)
        grayPlayableLetters = cv2.cvtColor(playable_letters, cv2.COLOR_BGR2GRAY)
        threshPlayableLetters = cv2.threshold(grayPlayableLetters, 40, 255, cv2.THRESH_BINARY)[1]
        stepX = threshPlayableLetters.shape[1]//7
        self.input_letters = '' 
        self.noWildcard = 0
        for x in range(7):
            startX = x * stepX
            endX = (x+1) * stepX
            cell = playable_letters[:, startX:endX]
            if self.determine_type_playable_letter(cell):
                whiteCell = threshPlayableLetters[:, startX:endX]
                whiteCell = cv2.resize(whiteCell, (78, 78), interpolation=cv2.INTER_LINEAR)
                whiteCell = whiteCell.reshape(1,78,78,1)/255.0
                letter=self.playableLetterLb.inverse_transform(self.playableLetterModel(whiteCell, training = False).numpy())[0][0]
                if letter == "4":
                    self.noWildcard +=1
                else:
                    self.input_letters += letter
    def solve(self):
        board = np.array(self.res)
        E = Engine()
        highscores = []
        list_of_letters = []
        if self.noWildcard==1:
            input_letters_copy = self.input_letters
            for extra in ALPHABET.keys():
                list_of_letters = []
                input_letters = input_letters_copy + extra
                for i in range(1, len(input_letters) + 1):
                    temp = list(permutations(input_letters, i))
                    temp = [list(perm) for perm in temp]
                    list_of_letters += temp
                for x in range(15):
                    for y in range(15):
                        if board[y,x].letter not in ALPHABET:
                            startPos = (x,y)
                            for DIR in ["HOR", "VER"]:
                                for letters in list_of_letters:
                                    squares = copy.copy(letters)
                                    squares = [Square(letter, True) if letter is not extra else Square(letter, False) for letter in squares]
                                    added_squares = E.calculate_added_squares(board, squares, startPos, DIR)
                                    added_letters = [(square[0].letter, (square[1])) for square in added_squares]
                                    if E.is_legal(board, added_letters, DIR):
                                        score = E.calculate_score(board, added_squares, DIR)
                                        if not highscores:
                                            heapq.heappush(highscores, (score, added_letters))
                                        else:
                                            if len(highscores) < self.capacity:
                                                if (score, added_letters) not in highscores:
                                                    heapq.heappush(highscores, (score, added_letters))
                                            else:
                                                if score > highscores[0][0] and (score, added_letters) not in highscores:
                                                    heapq.heappushpop(highscores, (score, added_letters))
        else:
            for i in range(1, len(self.input_letters) + 1):
                temp = list(permutations(self.input_letters, i))
                temp = [list(perm) for perm in temp]
                list_of_letters += temp
            for x in range(15):
                for y in range(15):
                    if board[y,x].letter not in ALPHABET:
                        startPos = (x,y)
                        for DIR in ["HOR", "VER"]:
                            for letters in list_of_letters:
                                squares = copy.copy(letters)
                                squares = [Square(letter, True) for letter in squares]
                                added_squares = E.calculate_added_squares(board, squares, startPos, DIR)
                                added_letters = [(square[0].letter, (square[1])) for square in added_squares]
                                if E.is_legal(board, added_letters, DIR):
                                    score = E.calculate_score(board, added_squares, DIR)
                                    if not highscores:
                                        heapq.heappush(highscores, (score, added_letters))
                                    else:
                                        if len(highscores) < self.capacity:
                                            if (score, added_letters) not in highscores:
                                                heapq.heappush(highscores, (score, added_letters))
                                        else:
                                            if score > highscores[0][0] and (score, added_letters) not in highscores:
                                                heapq.heappushpop(highscores, (score, added_letters))

        highscores = sorted(highscores, key = lambda x:x[0])[::-1]


        for index, (score, added_letters) in enumerate(highscores):
            background = copy.copy(self.image)
            for letter, position in added_letters:
                if self.VERSION == "IPHONE13PRO":
                    startX, endX = 606 + position[1]*78, 606 + (position[1]+1)*78
                    startY, endY = position[0]*78, (position[0]+1)*78
                elif self.VERSION == "IPHONE14PRO":
                    startX, endX = 626 + position[1]*78, 626 + (position[1]+1)*78
                    startY, endY = 5+position[0]*78, 5+(position[0]+1)*78
                overlay = cv2.imread(f'added_letters/{letter}_SC.jpg')
                background[startX:endX, startY:endY] = overlay
            word = ''.join([s[0] for s in added_letters])
            word = re.sub("1", "Á", word)
            word = re.sub("2", "Ä", word)
            word = re.sub("3", "Ö", word)
            background = cv2.putText(background, f'Place {word} for {score} points', ORG, FONT,  
                            FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)
            cv2.imwrite(f'{score}_{index}.png', background)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get the highest scoring words to place at the current position"
    )
    parser.add_argument("filename")
    parser.add_argument("--N", type=int, default=4)
    args = parser.parse_args()
    S = Solver(args.N)
    S.load_image_from_file(args.filename)
    S.scale_image()
    S.segment_image()
    img=S.solve()