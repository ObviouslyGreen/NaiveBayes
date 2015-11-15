import argparse
import logging
import numpy as np
import math
import matplotlib.pyplot as plt

from collections import Counter
from evaluation import calc_accuracy, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = './data'

class TextNaiveBayes:

    def __init__(self, type, runmode, k):
        # Model of different texts spam/normal email vs negative/positive review
        if type == 'spam_detection':
            self.type = 0
            self.classes = {'0': 0, '1': 1}
            self.class_names = ['normal email', 'spam']
            self.num_classes = 2
            self.train_file = '/train_email.txt'
            self.test_file = '/test_email.txt'
        elif type == 'movie_reviews':
            self.type = 1
            self.classes = {'-1': 0, '1': 1}
            self.class_names = ['negative review', 'positive review']
            self.num_classes = 2
            self.train_file = '/rt-train.txt'
            self.test_file = '/rt-test.txt'

        self.runmode = runmode
        self.k = k
        self.model = [{}, {}]
        self.word_counts = [0] * len(self.classes)
        self.doc_counts = np.zeros(self.num_classes)

    def train(self):
        with open(DATA_DIR + self.train_file) as f:
            for line in f:
                doc = line.split()
                model_class = self.classes[doc[0]]
                for x in doc[1:]:
                    x = x.split(':')
                    if x[0] in self.model[model_class]:
                        if(self.runmode == 'multinomial'):
                            self.model[model_class][x[0]] += int(x[1])
                        elif(self.runmode == 'bernoulli'):
                            self.model[model_class][x[0]] += 1
                    else:
                        if(self.runmode == 'multinomial'):
                            self.model[model_class][x[0]] = int(x[1])
                        elif self.runmode == 'bernoulli':
                            self.model[model_class][x[0]] = 1
                    for i in range(len(self.classes)):
                        if(i == model_class):
                            continue
                        if x[0] not in self.model[i]:
                            self.model[i][x[0]] = 0

                    self.word_counts[model_class] += int(x[1])
                self.doc_counts[model_class] += 1

        for x in range(len(self.classes)):
            for word in self.model[x]:
                self.model[x][word] += self.k
                if(self.runmode == 'multinomial'):
                    self.model[x][word] /= (self.word_counts[x] + len(self.model[x]) * self.k)
                elif(self.runmode == 'bernoulli'):
                    self.model[x][word] /= (self.doc_counts[x] + 2 * x)

        for model_class in range(self.num_classes):
            top_20 = []
            count = Counter(self.model[model_class])
            for word in count.most_common(20):
                top_20.append(word[0])
            logger.info('Top 20 words for class {0} are {1}'.format(self.class_names[model_class], top_20))

    def predict(self):
        correct_labels = []
        predicted_labels = []

        with open(DATA_DIR + self.test_file) as f:
            for line in f:
                doc = line.split()
                correct_labels.append(self.classes[doc[0]])
                map_classifier = np.zeros(self.num_classes)
                for model_class in range(self.num_classes):
                    map_classifier[model_class] = np.array(
                        [math.log(self.doc_counts[model_class]/np.sum(self.doc_counts))])
                    if self.runmode == 'multinomial':
                        for word in doc[1:]:
                            word = word.split(':')
                            if word[0] in self.model[model_class]:
                                map_classifier[model_class] += math.log(self.model[model_class][word[0]])
                    elif self.runmode == 'bernoulli':
                        words = [word.split(':')[0] for word in doc[1:]]
                        for word in self.model[model_class]:
                            if word in words:
                                map_classifier[model_class] += math.log(self.model[model_class][word])
                            else:
                                map_classifier[model_class] += math.log(1. - self.model[model_class][word])
                predicted_labels.append(np.argmax(map_classifier))

        logger.info((predicted_labels))
        correct_labels = np.array(correct_labels)
        predicted_labels = np.array(predicted_labels)
        accuracy = calc_accuracy(correct_labels, predicted_labels)
        logger.info('NB model is {0:.2f}% accurate on the {1} data with k = {2}.'
                    .format(accuracy, self.runmode, self.k))

        cm = confusion_matrix(correct_labels, predicted_labels, self.num_classes)

        plt.figure()
        plt.imshow(cm, cmap=plt.get_cmap('Greens'), interpolation='nearest')
        plt.xticks(np.arange(1))
        plt.yticks(np.arange(1))
        plt.xlabel('Predictions')
        plt.ylabel('Truths')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='''Text Document Naive Bayes Classifcation
                                                    for CS 440 by
                                                    Shibo Yao, Mike Chen,
                                                    and Jeff Zhu''')
    parser.add_argument('document_type', help='''Choose a type: spam_detection, movie_reviews''')
    parser.add_argument('runmode', help='''Choose a runmode: multinomial, bernoulli''')
    args = parser.parse_args()

    tnb = TextNaiveBayes(args.document_type, args.runmode, 1)
    tnb.train()
    tnb.predict()

if __name__ == '__main__':
    main()