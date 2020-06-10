import numpy as np

# Stacks the columns of a nXn matrix on top of other and makes a column


def matrix2col(matrix):
    col = np.zeros((0,), dtype='complex')
    for i in range(2):
        col = np.append(col, matrix[:, i])
    return col


def col2matrix(col):
    matrix = np.array([[col[0], col[2]], [col[1], col[3]]], dtype='complex')
    return matrix
