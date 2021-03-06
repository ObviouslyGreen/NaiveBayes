import argparse
import logging
import numpy as np
import math
import heapq
import matplotlib.pyplot as plt

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
        elif type == '8_newsgroups':
            self.type = 2
            self.classes = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7}
            self.class_names = ['sci.space', 'comp.sys.ibm.pc.hardware', 'rec.sport.baseball',
                                'comp.windows.x', 'talk.politics.misc', 'misc.forsale',
                                'rec.sport.hockey', 'comp.graphics']
            self.num_classes = 8
            self.train_file = '/8category.training.txt'
            self.test_file = '/8category.testing.txt'
        elif type == '20_newsgroups':
            self.type = 3
            self.classes = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                            '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12,
                            '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19}
            self.class_names = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
                                'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
                                'misc.forsale', 'rec.autos', 'rec.motorcycles',
                                'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
                                'sci.electronics', 'sci.med', 'sci.space',
                                'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
                                 'talk.politics.misc', 'talk.religion.misc']
            self.num_classes = 20
            self.train_file = '/train.data.txt'
            self.train_label = '/train.label.txt'
            self.test_label = '/test.label.txt'
            self.test_file = '/test.data.txt'

        self.runmode = runmode
        self.k = k
        self.model = {}
        self.word_counts = [0] * len(self.classes)
        self.doc_counts = np.zeros(self.num_classes)
        self.vocab_list = {}
        self.document_list = {}
        self.document_test_list = {}

    def train(self):
        if self.type != 3:
            with open(DATA_DIR + self.train_file) as f:
                for line in f:
                    doc = line.split()
                    model_class = self.classes[doc[0]]
                    for x in doc[1:]:
                        x = x.split(':')

                        # Increment if in vocabulary
                        if x[0] in self.model:
                            if(self.runmode == 'multinomial'):
                                self.model[x[0]][model_class] += int(x[1])
                            elif(self.runmode == 'bernoulli'):
                                self.model[x[0]][model_class] += 1
                        # Initialize list and count for new word in vocabulary
                        else:
                            if(self.runmode == 'multinomial'):
                                self.model[x[0]] = [0] * self.num_classes
                                self.model[x[0]][model_class] = int(x[1])
                            elif self.runmode == 'bernoulli':
                                self.model[x[0]] = [0] * self.num_classes
                                self.model[x[0]][model_class] = 1

                        self.word_counts[model_class] += int(x[1])
                    self.doc_counts[model_class] += 1

            for word in self.model:
                for x in range(self.num_classes):
                    self.model[word][x] += self.k
                    if(self.runmode == 'multinomial'):
                        self.model[word][x] /= (self.word_counts[x] + len(self.model) * self.k)
                    elif(self.runmode == 'bernoulli'):
                        self.model[word][x] /= (self.doc_counts[x] + 2 * self.k)

            for model_class in range(self.num_classes):
                top_20 = heapq.nlargest(20, self.model.items(), lambda k: k[1][model_class])
                logger.info('Top 20 words for class {0} are {1}'.
                            format(self.class_names[model_class], [word[0] for word in top_20]))
        else:   # 20_group~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ train
            # Initialize vocab list and count for new word in vocabulary
            with open(DATA_DIR + '/matlab/vocabulary.txt') as h:
                for idx, word in enumerate(h):
                    self.vocab_list[idx+1] = word.rstrip()           # vocab dict w/line number as index
            with open(DATA_DIR + '/matlab'+ self.train_label) as g:
                for idx_label, label_line in enumerate(g):  # label dict w/line number as index
                    self.document_list[idx_label+1] = label_line.rstrip()
            with open(DATA_DIR + '/matlab'+ self.train_file) as f:
                # doc_data[0] is label line number
                # doc_data[1] is vocab list line number
                # doc_data[2] is amount of that word
                last_doc = 0
                for data_line in f:
                    doc_data = data_line.split()
                    label_line = int(doc_data[0])
                    doc_label = self.document_list[label_line]
                    model_class = self.classes[str(doc_label).rstrip()]
                    curr_word = self.vocab_list[int(doc_data[1])]
                    # Increment if in vocabulary
                    if curr_word in self.model:
                        if self.runmode == 'multinomial':
                            self.model[curr_word][model_class] += int(doc_data[2])
                        elif self.runmode == 'bernoulli':
                            self.model[curr_word][model_class] += 1
                    # Initialize list and count for new word in vocabulary
                    else:
                        if self.runmode == 'multinomial':
                            self.model[curr_word] = [0] * self.num_classes
                            self.model[curr_word][model_class] = int(doc_data[2])
                        elif self.runmode == 'bernoulli':
                            self.model[curr_word] = [0] * self.num_classes
                            self.model[curr_word][model_class] = 1

                    self.word_counts[model_class] += int(doc_data[2])
                    if label_line != last_doc:
                        self.doc_counts[model_class] += 1
                        last_doc = label_line

            for word in self.model:
                for x in range(self.num_classes):
                    self.model[word][x] += self.k
                    if(self.runmode == 'multinomial'):
                        self.model[word][x] /= (self.word_counts[x] + len(self.model) * self.k)
                    elif(self.runmode == 'bernoulli'):
                        self.model[word][x] /= (self.doc_counts[x] + 2 * self.k)
            for model_class in range(self.num_classes):
                top_20 = heapq.nlargest(20, self.model.items(), lambda k: k[1][model_class])
                logger.info('Top 20 words for class {0} are {1}'.
                            format(self.class_names[model_class], [word[0] for word in top_20]))



    def predict(self):
        if self.type != 3:
            correct_labels = []
            predicted_labels = []
            counter = 0
            with open(DATA_DIR + self.test_file) as f:
                for line in f:
                    counter += 1
                    doc = line.split()
                    correct_labels.append(self.classes[doc[0]])
                    map_classifier = np.zeros(self.num_classes)
                    for model_class in range(self.num_classes):
                        map_classifier[model_class] = np.array(
                            [math.log(float(self.doc_counts[model_class]/np.sum(self.doc_counts)))])
                        if self.runmode == 'multinomial':
                            for word in doc[1:]:
                                word = word.split(':')
                                if word[0] in self.model:
                                    map_classifier[model_class] += math.log(self.model[word[0]][model_class])
                        elif self.runmode == 'bernoulli':
                            words = [word.split(':')[0] for word in doc[1:]]
                            for word in self.model:
                                if word in words:
                                    map_classifier[model_class] += math.log(self.model[word][model_class])
                                else:
                                    map_classifier[model_class] += math.log(1. - self.model[word][model_class])
                    predicted_labels.append(np.argmax(map_classifier))
                    logger.info('{0}'.format(counter))
        else:
            correct_labels = []
            predicted_labels = []
            counter = 0
            with open(DATA_DIR + '/matlab'+ self.test_label) as g:
                for idx_label, label_line in enumerate(g):      # label dict w/line number as index
                    self.document_test_list[idx_label+1] = int(label_line.rstrip())
                    correct_labels.append(int(label_line.rstrip()))     # correct
            with open(DATA_DIR + '/matlab' + self.test_file) as f:
                last_doc = 0
                if self.runmode == 'multinomial':
                    for data_line in f:
                        doc_data = data_line.split()
                        if (int(doc_data[0]) != last_doc) & (last_doc != 0):
                           # counter += 1
                            predicted_labels.append(np.argmax(map_classifier))
                           # logger.info('{0}'.format(counter))
                        if (int(doc_data[0]) != last_doc) & (int(doc_data[0]) != ''):
                            map_classifier = np.zeros(self.num_classes)
                            last_doc = int(doc_data[0])
                            for model_class in range(self.num_classes):     # setup new mapclassifier after predict
                                map_classifier[model_class] = np.array(
                                    [math.log(float(self.doc_counts[model_class]/np.sum(self.doc_counts)))])

                        for model_class in range(self.num_classes):
                            curr_word = self.vocab_list[int(doc_data[1])]
                            if curr_word in self.model:
                                map_classifier[model_class] += math.log(self.model[curr_word][model_class])
                    predicted_labels.append(np.argmax(map_classifier))
                elif self.runmode == 'bernoulli':
                    for idx, data_line in enumerate(f):
                        doc_data = data_line.split()
                       # if (int(doc_data) != last_doc) & (doc_data != ''):

        correct_labels = np.array(correct_labels)
        predicted_labels = np.array(predicted_labels)
        accuracy = calc_accuracy(correct_labels, predicted_labels)
        logger.info('NB model is {0:.2f}% accurate on the {1} data with k = {2}.'
                    .format(accuracy, self.runmode, self.k))

        cm = confusion_matrix(correct_labels, predicted_labels, self.num_classes)
        class_accuracies = [cm[n][n] for n in range(self.num_classes)]
        for n, x in enumerate(class_accuracies):
            logger.info('Class {0} has an accuracy of {1:.2f}%'.format(self.class_names[n], 100 * x))

        plt.figure(figsize=(30,30))
        plt.imshow(cm, cmap=plt.get_cmap('Greens'), interpolation='nearest')
        plt.xticks(np.arange(self.num_classes), self.class_names, fontsize = 5)
        plt.yticks(np.arange(self.num_classes), self.class_names, fontsize = 10)
        plt.xlabel('Predictions')
        plt.ylabel('Truths')
        plt.colorbar()
        plt.show()
            # 20_group~~~~~~~~~~~~~~~~~~ predict



def main():
    parser = argparse.ArgumentParser(description='''Text Document Naive Bayes Classifcation
                                                    for CS 440 by
                                                    Shibo Yao, Mike Chen,
                                                    and Jeff Zhu''')
    parser.add_argument('document_type', help='''Choose a type: spam_detection, movie_reviews, 8_newsgroups''')
    parser.add_argument('runmode', help='''Choose a runmode: multinomial, bernoulli''')
    parser.add_argument('k_value', help='''Choose a k value for Laplacian smoothing''')

    args = parser.parse_args()

    tnb = TextNaiveBayes(args.document_type, args.runmode, float(args.k_value),)
    tnb.train()
    tnb.predict()

if __name__ == '__main__':
    main()