import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = './data'

class TextNaiveBayes:

    def __init__(self, type):
        # Model of different texts spam/normal email vs negative/positive review
        if type == 'spam_detection':
            self.type = 0
            self.classes = {'0': 0, '1': 1}
            self.train_file = '/train_email.txt'
            self.test_file = '/test_email.txt'
        elif type == 'movie_reviews':
            self.type = 1
            self.classes = {'-1': 0, '1': 1}
            self.train_file = '/rt-train.txt'
            self.test_file = '/rt-test.txt'

        self.model = [{}, {}]
        self.word_counts = [0] * len(self.classes)

    def _train_multinomial(self, k):
        with open(DATA_DIR + self.train_file) as f:
            for line in f:
                doc = line.split()
                for x in doc[1:]:
                    model_class = self.classes[doc[0]]
                    x = x.split(':')
                    if x[0] in self.model[model_class]:
                        self.model[model_class][x[0]] += int(x[1])
                    else:
                        self.model[model_class][x[0]] = int(x[1])
                    self.word_counts[model_class] += int(x[1])

        logger.info(self.model[0])
        for x in range(len(self.classes)):
            for word in self.model[x]:
                self.model[x][word] += k
                self.model[x][word] /= (self.word_counts[x] + len(self.model[x]) * k)
        logger.info(self.model[0])
        logger.info(self.word_counts)

    def _train_bernoulli(self, k):
        return

    def train(self, k, runmode):
        if runmode == 'multinomial':
            self._train_multinomial(k)
        elif runmode == 'bernoulli':
            self._train_bernoulli(k)

def main():
    parser = argparse.ArgumentParser(description='''Text Document Naive Bayes Classifcation
                                                    for CS 440 by
                                                    Shibo Yao, Mike Chen,
                                                    and Jeff Zhu''')
    parser.add_argument('document_type', help='''Choose a type: spam_detection, movie_reviews''')
    parser.add_argument('runmode', help='''Choose a runmode: multinomial, bernoulli''')
    args = parser.parse_args()

    tnb = TextNaiveBayes(args.document_type)
    tnb.train(2, args.runmode)


if __name__ == '__main__':
    main()