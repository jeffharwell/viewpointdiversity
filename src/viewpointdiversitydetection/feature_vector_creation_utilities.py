"""
Module containing various functions I have found useful in assembling the feature vector.
"""

import numpy as np


def combine_as_max(vector1, vector2):
    """
    Combine two vectors and return a vector that has the maximum values from each vector compared pairwise.

    :param vector1: First list to compare
    :param vector2: Second list to compare
    :return: a list containing the max of each element of vector1 and vector2 compared pairwise.
    """
    combined_vector = []
    if len(vector1) != len(vector2):
        raise RuntimeError("Vectors must be of equal length!")
    for index in range(0, len(vector1)):
        if vector1[index] > vector2[index]:
            combined_vector.append(vector1[index])
        else:
            combined_vector.append(vector2[index])
    return combined_vector


def combine_by_append(vector1, vector2):
    """
    Combines two lists by appending them.

    :param vector1: list of values
    :param vector2: list of values
    :return: a list created by appending vector1 to vector2
    """
    return vector1 + vector2


def combine_as_average(vector1, vector2, include_zeros=True):
    """
    Takes the average of two vectors. It can include zeros when computing the average or drop them.

    :param vector1: list of values
    :param vector2: list of values
    :param include_zeros: boolean, if True zeros will be included in the average, if False they will be dropped.
    :return: list containing each element is the average of the elements from the two vectors
    """

    if len(vector1) != len(vector2):
        raise RuntimeError("Vectors must be of equal length!")
    empty_vector = [0.0] * len(vector1)
    if vector1 == empty_vector and vector2 == empty_vector:
        # We received two zero vectors, don't need to compute an average for that
        return empty_vector
    if vector1 == empty_vector and not include_zeros:
        # If vector1 is empty, and we are not including zeros, then the average is vector2
        return vector2
    if vector2 == empty_vector and not include_zeros:
        # If vector2 is empty, and we are not including zeros, then the average is vector1
        return vector1
    nv1 = np.array(vector1)
    nv2 = np.array(vector2)
    avg = np.divide(np.sum([nv1, nv2], axis=0), 2.0)
    return list(avg)


def combine_as_average_np(vector1, vector2, include_zeros=True):
    """
    Combine two numpy vectors (numpy array of values) by averaging them.

    :param vector1: first numpy array
    :param vector2: second numpy array
    :param include_zeros: include_zeros: boolean, if True zeros will be included in the average,
                          if False they will be dropped.
    :return: a numpy array, each element is the average of the elements from the two vectors
    """
    if len(vector1) != len(vector2):
        raise RuntimeError("Vectors must be of equal length!")
    empty_sentiment_vector = np.array([0.0] * len(vector1))
    if np.array_equal(vector1, empty_sentiment_vector) and np.array_equal(vector2, empty_sentiment_vector):
        # We received two zero vectors, don't need to compute an average for that
        return empty_sentiment_vector
    if np.array_equal(vector1, empty_sentiment_vector) and not include_zeros:
        # If vector1 is empty, and we are not including zeros, then the average is vector2
        return vector2
    if np.array_equal(vector2, empty_sentiment_vector) and not include_zeros:
        # If vector2 is empty, and we are not including zeros, then the average is vector1
        return vector1
    avg = np.divide(np.sum([vector1, vector2], axis=0), 2.0)
    return avg


def combine_as_average_with_boost(vector1, boost, vector2):
    """
    Combine two vectors but "boost" the first vector by multiplying every element by a number.

    :param vector1: list of values
    :param boost: multiplier to be applied to the first list of values
    :param vector2: list of values
    :return: A list of values, each element is the
    """
    combined_vector = []
    if len(vector1) != len(vector2):
        raise RuntimeError("Vectors must be of equal length!")
    for index in range(0, len(vector1)):
        avg = (vector1[index]*boost + vector2[index])/(2 + boost - 1)
        combined_vector.append(avg)
    return combined_vector


def create_has_sentiments_present_vector(vector1, vector2):
    """
    Create a short vector. If the the vectors are not equal
    to zero it returns [1,1], if either of the vectors are equal
    to zero it will return a zero for that vector. For example, if
    vector1 is a zero vector, but vector2 has values, the function
    would return [0,1]

    :param vector1: list of values
    :param vector2: list of values
    :return: A list with two values; indicating a 0 if the vector is 0, a one otherwise.
    """
    if len(vector1) != len(vector2):
        raise RuntimeError("Vectors must be of equal length!")
    empty_sentiment_vector = [0.0] * len(vector1)
    present_vector = [0, 0]
    if vector1 != empty_sentiment_vector:
        present_vector[0] = 1
    if vector2 != empty_sentiment_vector:
        present_vector[1] = 1
    return present_vector


def create_word2vec_present_vector(vector1, vector2):
    """
    Create a short vector. If the the vectors are not equal
    to zero it returns [1,1], if either of the vectors are equal
    to zero it will return a zero for that vector. For example, if
    vector1 is a zero vector, but vector2 has values, the function
    would return [0,1]

    :param vector1: list of values
    :param vector2: list of values
    :return: A list with two values; indicating a 0 if the vector is 0, a one otherwise.
    """
    if len(vector1) != len(vector2):
        raise RuntimeError("Vectors must be of equal length!")
    empty_sentiment_vector = np.zeros(len(vector1))
    present_vector = [0, 0]
    if not np.array_equal(vector1, empty_sentiment_vector):
        present_vector[0] = 1
    if not np.array_equal(vector2, empty_sentiment_vector):
        present_vector[1] = 1
    return present_vector
