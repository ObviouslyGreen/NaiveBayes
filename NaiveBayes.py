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

    def train(self):
        training_data = np.chararray((10, SIZE, SIZE), unicode=True)
        with open(DATA_DIR + '/traininglabels') as f0, open(DATA_DIR + '/trainingimages') as f1:
            for line in f0:
                curr_num = int(line)
                for row in range(SIZE):
                    training_data[curr_num][row] = np.array(list(f1.readline().rstrip('\n')))


def main():
    nb = NaiveBayes()
    nb.train()


if __name__ == '__main__':
    main()