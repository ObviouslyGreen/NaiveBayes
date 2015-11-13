import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SIZE = 28
DATA_DIR = './data'


class NaiveBayes:

    def __init__(self):
        # Model of 10 digits with images of (size, size)
        self.model = np.zeros((10, SIZE, SIZE))

    def train(self):
        with open(DATA_DIR + '/traininglabels') as f0, open(DATA_DIR + '/trainingimages') as f1:
            for line in f0:
                curr_num = int(line)
                for row in range(SIZE):
                    curr_img_row = list(f1.readline().rstrip('\n'))
                    for col in curr_img_row:
                        # do training stuff here


def main():
    nb = NaiveBayes()
    nb.train()


if __name__ == '__main__':
    main()