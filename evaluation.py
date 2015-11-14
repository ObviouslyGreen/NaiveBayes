import numpy as np
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calc_accuracy(truths, predictions):
    mispredict_count = np.count_nonzero(truths - predictions)
    return 100 * (len(truths) - mispredict_count) / len(truths)


def confusion_matrix(truths, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes))
    for x, y in zip(truths, predictions):
        matrix[x][y] += 1
    row_sums = matrix.sum(axis=1)
    return matrix / row_sums


def main():
        truths = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        predictions = np.array([1, 20, 3, -5, 5, 6, 0, 8, 9, 10])
        np.testing.assert_equal(calc_accuracy(truths, predictions), 70.0)

        truths = np.array([0, 1, 2, 3])
        predictions = np.array([0, 1, 3, 2])
        cm = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        np.testing.assert_array_equal(confusion_matrix(truths, predictions, 4), cm)

        logger.info('All tests passed!')


if __name__ == '__main__':
    main()