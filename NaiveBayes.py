import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SIZE = 28
DATA_DIR = './data'


class NaiveBayes:

    def __init__(self):
        # Model of 10 digits
        # Digits are images with values ' ', '+', or '#'
        self.model = np.zeros((10, SIZE, SIZE, 3))
        self.num_count = np.zeros(10)


    def train(self, k):
        with open(DATA_DIR + '/traininglabels') as f0, open(DATA_DIR + '/trainingimages') as f1:
            for line in f0:
                curr_num = int(line)
                self.num_count[curr_num] += 1
                for row in range(SIZE):
                    img_row = (f1.readline().rstrip('\n'))
                    for col in range(SIZE):
                        if img_row[col] ==  ' ':
                            self.model[curr_num][row][col][0] += 1
                        elif img_row[col] == '+':
                            self.model[curr_num][row][col][1] += 1
                        elif img_row[col] == '#':
                            self.model[curr_num][row][col][2] += 1

        for num in range(10):
            for row in range(SIZE):
                for col in range(SIZE):
                    self.model[num][row][col] += k
                    self.model[num][row][col] /= (self.num_count[num] + 3 * k)

def main():
    nb = NaiveBayes()
    nb.train(25)



if __name__ == '__main__':
    main()