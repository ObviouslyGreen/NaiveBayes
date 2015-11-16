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
        if not num_features or num_features == 2 or (num_features == 3 and self.runmode == 'digits'):
            self.num_features = num_features
        else:
            raise ValueError('Invalid number of features.')
        if not k or 1 <= k <= 50:
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
            self.num_counts = np.zeros(self.num_classes)
        if num_features:
            self.num_features = num_features
            self.num_counts = np.zeros(self.num_classes)

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
                    self.model[num][y][x] /= (self.num_counts[num] + self.num_features * self.k)

        logger.info('Finished training in {0:.2f} seconds'.format(time.time() - start_time))

    def predict(self, info=True):
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
            predicted_label = np.argmax(map_classifier)
            predicted_labels.append((predicted_label, map_classifier[predicted_label], n))

        truths = np.array(correct_labels)
        predictions = np.array([x[0] for x in predicted_labels])
        accuracy = calc_accuracy(truths, predictions)
        logger.info('NB model is {0:.2f}% accurate on the {1} data with k = {2}.'.format(accuracy, self.runmode, self.k))

        if info:
            cm = confusion_matrix(truths, predictions, self.num_classes)
            class_accuracies = [cm[n][n] for n in range(self.num_classes)]
            # Class accuracies
            for n, x in enumerate(class_accuracies):
                logger.info('Class {0} has an accuracy of {1:.2f}%'.format(n, x))

            # Confusion matrix
            plt.figure()
            plt.imshow(cm, cmap=plt.get_cmap('Greens'), interpolation='nearest')
            plt.title('Confusion Matrix')
            plt.xticks(np.arange(self.num_classes))
            plt.yticks(np.arange(self.num_classes))
            plt.xlabel('Predictions')
            plt.ylabel('Truths')

            # Test images with the highest and lowest posterior probability
            # Sorts from lowest to highest by class, then by posterior probability
            sorted_predictions = sorted(predicted_labels)
            class_indices = []
            for x in range(len(sorted_predictions)):
                if sorted_predictions[x][0] != sorted_predictions[x-1][0]:
                    class_indices.append(x)

            for x in range(len(class_indices)):
                curr_class = sorted_predictions[class_indices[x]][0]
                lowest_idex = sorted_predictions[class_indices[x]][2]
                try:
                    highest_idx = sorted_predictions[class_indices[x+1]-1][2]
                except IndexError:
                    highest_idx = sorted_predictions[len(sorted_predictions)-1][2]
                best_test_image = [[0 if x in ['#', '+'] else 1 for x in y] for y in test_images[highest_idx]]
                worst_test_image = [[0 if x in ['#', '+'] else 1 for x in y] for y in test_images[lowest_idex]]
                plt.figure()
                plt.suptitle('Class {0}'.format(curr_class))
                plt.subplot(1, 2, 1)
                plt.imshow(best_test_image, cmap=plt.get_cmap('Greys_r'))
                plt.title('Highest')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(1, 2, 2)
                plt.title('Lowest')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(worst_test_image, cmap=plt.get_cmap('Greys_r'))

            # Odds ratio for the four worst classes
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
                    plt.title('Likelihood of {0}'.format(i))
                    plt.xticks([])
                    plt.yticks([])
                    cbar = plt.colorbar(shrink=0.35)
                    cbar.set_ticks(np.arange(np.amin(log_likelihood_one), np.amax(log_likelihood_one), step=2, dtype=np.int8))
                    for t in cbar.ax.get_yticklabels():
                        t.set_horizontalalignment('right')
                        t.set_x(4)
                    plt.subplot(1, 3, 2)
                    plt.imshow(log_likelihood_two, interpolation='nearest')
                    plt.title('Likelihood of {0}'.format(j))
                    plt.xticks([])
                    plt.yticks([])
                    cbar = plt.colorbar(shrink=0.35)
                    cbar.set_ticks(np.arange(np.amin(log_likelihood_two), np.amax(log_likelihood_two), step=2, dtype=np.int8))
                    for t in cbar.ax.get_yticklabels():
                        t.set_horizontalalignment('right')
                        t.set_x(4)
                    plt.subplot(1, 3, 3)
                    plt.imshow(odds_ratio, interpolation='nearest')
                    plt.title('Odds ratio')
                    plt.xticks([])
                    plt.yticks([])
                    cbar = plt.colorbar(shrink=0.35)
                    cbar.set_ticks(np.arange(np.amin(odds_ratio), np.amax(odds_ratio), step=2, dtype=np.int8))
                    for t in cbar.ax.get_yticklabels():
                        t.set_horizontalalignment('right')
                        t.set_x(4)

            plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runmode', help='Determine which dataset to use: digits or faces')
    parser.add_argument('-k', type=int, help='Smoothing factor')
    parser.add_argument('-f', '--num_features', type=int)
    args = parser.parse_args()

    dnb = DigitNaiveBayes(args.runmode, args.num_features, args.k)
    if args.k:
        dnb.train()
        dnb.predict()
    else:
        for x in range(1, 51):
            dnb.train(k=x)
            dnb.predict(info=False)


if __name__ == '__main__':
    main()
