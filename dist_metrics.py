import math
import random
import csv
import cProfile
import numpy as np
import hashlib


def manhattan_distance(a,b):
    return np.sum(np.fabs(a,b))

def euclidean_distance(a,b):
    return np.linalg.norm(a,b)


def cosine_similarity(a,b):

    ab = np.dot(a,b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    return 1 - (ab / (a_norm * b_norm))


def tanimoto_coefficient(a,b):

    ab = np.dot(a,b)
    a_square = np.dot(a,a)
    b_square = np.dot(b,b)
    return 1 - (ab / (a_square + b_square - ab))







