import argparse
import logging
import math
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_DIR = './data'


class DigitNaiveBayes:

    def __init__(self, runmode='digits', num_features=3, k=1):
        self.runmode = runmode
        if runmode == 'digits':
            self.num_classes = 10
            self.size = 28
        if num_features in [2, 3]:
            self.num_features = num_features
        else:
            raise ValueError('Number of features can only be 2 or 3')
        if 1 <= k <= 50:
            self.k = k
        else:
            raise ValueError('Smoothing factor can only be from 1 to 50')
        # Images with pixel values ' ', '+', or '#'
        self.model = np.zeros((self.num_classes, self.size, self.size, num_features))
        self.num_counts = np.zeros(self.num_classes)

    def train(self, k=None, num_features=None):
        if self.runmode == 'digits':
            training_label_path = DATA_DIR + '/traininglabels'
            training_images_path = DATA_DIR + '/trainingimages'
        if k:
            self.k = k
        if num_features:
            self.num_features = num_features

        with open(training_label_path) as f0, open(training_images_path) as f1:
            for line in f0:
                curr_num = int(line)
                self.num_counts[curr_num] += 1
                for row in range(self.size):
                    img_row = f1.readline().rstrip('\n')
                    for col in range(self.size):
                        if img_row[col] ==  ' ':
                            self.model[curr_num][row][col][0] += 1
                        elif img_row[col] == '+':
                            self.model[curr_num][row][col][1] += 1
                        elif img_row[col] == '#':
                            self.model[curr_num][row][col][2] += 1

        for num in range(self.num_classes):
            for row in range(self.size):
                for col in range(self.size):
                    self.model[num][row][col] += self.k
                    self.model[num][row][col] /= (self.num_counts[num] + 3 * self.k)

    def predict(self):
        if self.runmode == 'digits':
            test_label_path = DATA_DIR + '/testlabels'
            test_images_path = DATA_DIR + '/testimages'

        correct_labels = []

        with open(test_label_path) as f:
            for line in f:
                correct_labels.append(int(line))

        num_images = len(correct_labels)
        # Using python list instead of np since np chararrays replace spaces with empty string
        test_images = [[None for _ in range(self.size)] for _ in range(num_images)]

        with open(test_images_path) as f:
            for n in range(num_images):
                for row in range(self.size):
                    test_images[n][row] = list(f.readline().rstrip('\n'))

        predicted_labels = []
        for n in range(num_images):
            map_classifier = np.zeros(self.num_classes)
            for num in range(self.num_classes):
                map_classifier[num] = np.array([math.log(self.num_counts[num]/np.sum(self.num_counts))])
                for row in range(self.size):
                    for col in range(self.size):
                        if test_images[n][row][col] == ' ':
                            map_classifier[num] += math.log(self.model[num][row][col][0])
                        elif test_images[n][row][col] == '+':
                            map_classifier[num] += math.log(self.model[num][row][col][1])
                        elif test_images[n][row][col] == '#':
                            map_classifier[num] += math.log(self.model[num][row][col][2])
            predicted_labels.append(np.argmax(map_classifier))

        correct_labels = np.array(correct_labels)
        predicted_labels = np.array(predicted_labels)
        num_mispredict = np.count_nonzero(correct_labels - predicted_labels)
        accuracy = (num_images - num_mispredict) / num_images
        logger.info('Model is {0:.2f}%% accurate with k = {1}'.format(accuracy, self.k))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runmode', help='Determine which dataset to use')
    parser.add_argument('-k', type=int, help='Smoothing factor')
    parser.add_argument('-f', '--num_features', type=int)
    args = parser.parse_args()

    dnb = DigitNaiveBayes(args.runmode, args.num_features, args.k)
    dnb.train()
    dnb.predict()


if __name__ == '__main__':
    main()
