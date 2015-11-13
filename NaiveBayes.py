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
        pass


def main():
    nb = NaiveBayes()
    nb.train()


if __name__ == '__main__':
    main()