import argparse
import heapq
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import operator
import time

from evaluation import calc_accuracy, confusion_matrix


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = './data'


class DigitNaiveBayes:

    def __init__(self, runmode='digits', num_features=3, k=1):
        self.runmode = runmode
        if runmode == 'digits':
            self.num_classes = 10
            self.col = 28
            self.row = 28
        elif runmode == 'faces':
            self.num_classes = 2
            self.col = 60
            self.row = 70
        else:
            raise ValueError('Invalid runmode.')
        if num_features == 2 or (num_features == 3 and self.runmode == 'digits'):
            self.num_features = num_features
        else:
            raise ValueError('Invalid number of features.')
        if 1 <= k <= 50:
            self.k = k
        else:
            raise ValueError('Invalid smoothing number.')
        # Images with pixel values ' ', '+', or '#'
        self.model = np.zeros((self.num_classes, self.row, self.col, num_features))
        self.num_counts = np.zeros(self.num_classes)

    def train(self, k=None, num_features=None):
        if self.runmode == 'digits':
            training_label_path = DATA_DIR + '/traininglabels'
            training_images_path = DATA_DIR + '/trainingimages'
        elif self.runmode == 'faces':
            training_label_path = DATA_DIR + '/facedatatrainlabels'
            training_images_path = DATA_DIR + '/facedatatrain'
        if k:
            self.k = k
        if num_features:
            self.num_features = num_features

        start_time = time.time()

        with open(training_label_path) as f0, open(training_images_path) as f1:
            for line in f0:
                curr_num = int(line)
                self.num_counts[curr_num] += 1
                for y in range(self.row):
                    img_row = f1.readline().rstrip('\n')
                    for x in range(self.col):
                        if img_row[x] ==  ' ':
                            self.model[curr_num][y][x][0] += 1
                        if self.num_features == 3:
                            if img_row[x] == '+':
                                self.model[curr_num][y][x][1] += 1
                            elif img_row[x] == '#':
                                self.model[curr_num][y][x][2] += 1
                        else:
                            if img_row[x] in ['+', '#']:
                                self.model[curr_num][y][x][1] += 1

        for num in range(self.num_classes):
            for x in range(self.col):
                for y in range(self.row):
                    self.model[num][y][x] += self.k
                    self.model[num][y][x] /= (self.num_counts[num] + 3 * self.k)

        logger.info('Finished training.')
        logger.info('Training took {0:.2f} seconds'.format(time.time() - start_time))

    def predict(self):
        if self.runmode == 'digits':
            test_label_path = DATA_DIR + '/testlabels'
            test_images_path = DATA_DIR + '/testimages'
        elif self.runmode == 'faces':
            test_label_path = DATA_DIR + '/facedatatestlabels'
            test_images_path = DATA_DIR + '/facedatatest'

        correct_labels = []

        with open(test_label_path) as f:
            for line in f:
                correct_labels.append(int(line))

        num_images = len(correct_labels)
        # Using python list instead of np since np chararrays replace spaces with empty string
        test_images = [[None for _ in range(self.row)] for _ in range(num_images)]

        with open(test_images_path) as f:
            for n in range(num_images):
                for y in range(self.row):
                    test_images[n][y] = list(f.readline().rstrip('\n'))

        predicted_labels = []
        for n in range(num_images):
            map_classifier = np.zeros(self.num_classes)
            for num in range(self.num_classes):
                map_classifier[num] = np.array([math.log(self.num_counts[num]/np.sum(self.num_counts))])
                for y in range(self.row):
                    for x in range(self.col):
                        if test_images[n][y][x] == ' ':
                            map_classifier[num] += math.log(self.model[num][y][x][0])
                        if self.num_features == 3:
                            if test_images[n][y][x] == '+':
                                map_classifier[num] += math.log(self.model[num][y][x][1])
                            elif test_images[n][y][x] == '#':
                                map_classifier[num] += math.log(self.model[num][y][x][2])
                        else:
                            if test_images[n][y][x] in ['+', '#']:
                                map_classifier[num] += math.log(self.model[num][y][x][1])
            predicted_labels.append(np.argmax(map_classifier))

        correct_labels = np.array(correct_labels)
        predicted_labels = np.array(predicted_labels)
        accuracy = calc_accuracy(correct_labels, predicted_labels)
        logger.info('NB model is {0:.2f}% accurate on the {1} data with k = {2}.'.format(accuracy, self.runmode, self.k))

        cm = confusion_matrix(correct_labels, predicted_labels, self.num_classes)
        class_accuracies = [cm[n][n] for n in range(self.num_classes)]

        max_n = np.argmax(np.array(class_accuracies))
        min_n = np.argmin(np.array(class_accuracies))
        logger.info('Class {0} has the highest posterior probability with an accuracy of {1:.2f}%.'.format(max_n, 100 * cm[max_n][max_n]))
        logger.info('Class {0} has the highest posterior probability with an accuracy of {1:.2f}%.'.format(min_n, 100 * cm[min_n][min_n]))

        plt.figure()
        plt.imshow(cm, cmap=plt.get_cmap('Greens'), interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.xticks(np.arange(self.num_classes))
        plt.yticks(np.arange(self.num_classes))
        plt.xlabel('Predictions')
        plt.ylabel('Truths')

        cm_ravel = np.ravel(cm)
        least_accurate_pairs = cm_ravel.argsort()[:4]
        least_accurate_pairs = [(x % self.num_classes, math.floor(x / self.num_classes)) for x in least_accurate_pairs]

        if self.num_features == 2 and self.runmode == 'digits':
            for i, j in least_accurate_pairs:
                log_likelihood_one = np.zeros((self.col, self.row))
                log_likelihood_two = np.zeros((self.col, self.row))
                odds_ratio = np.zeros((self.col, self.row))
                for y in range(self.row):
                    for x in range(self.col):
                        log_likelihood_one[y][x] = math.log(self.model[i][y][x][1])
                        log_likelihood_two[y][x] = math.log(self.model[j][y][x][1])
                        odds_ratio[y][x] = math.log(self.model[i][y][x][1] / self.model[j][y][x][1])

                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(log_likelihood_one, interpolation='nearest')
                plt.title('Log likelihood of {0}'.format(i))
                plt.xticks([])
                plt.yticks([])
                plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.imshow(log_likelihood_two, interpolation='nearest')
                plt.title('Log likelihood of {0}'.format(j))
                plt.xticks([])
                plt.yticks([])
                plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.imshow(odds_ratio, interpolation='nearest')
                plt.title('Odds ratio between {0} and {1}'.format(i, j))
                plt.xticks([])
                plt.yticks([])
                plt.colorbar()

        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runmode', help='Determine which dataset to use: digits or faces')
    parser.add_argument('-k', type=int, help='Smoothing factor')
    parser.add_argument('-f', '--num_features', type=int)
    args = parser.parse_args()

    dnb = DigitNaiveBayes(args.runmode, args.num_features, args.k)
    dnb.train()
    dnb.predict()


if __name__ == '__main__':
    main()
