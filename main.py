import numpy as np
import os
import glob
import csv

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2

script_dir = os.path.dirname(__file__)

def get_data(files):
    print(files)

    data_matrix = np.loadtxt(files[0], dtype='i', delimiter='\t') #loading data from first textfile
    data_matrix = data_matrix.T #transpond data_matrix array

    last_col = [0] * len(data_matrix)
    data_matrix = np.column_stack((data_matrix, last_col)) #adding one column to data_matrix

    for x in range(len(files) - 1):
        temp_matrix = np.loadtxt(files[x + 1], dtype='i', delimiter='\t') #loading data from textfile x
        temp_matrix = temp_matrix.T #transpod array
        last_col = [x+1] * len(temp_matrix)
        temp_matrix = np.column_stack((temp_matrix, last_col))
        data_matrix = np.concatenate((data_matrix, temp_matrix), axis=0) #join new matrix to the rest

    X = data_matrix[:, :-1]
    Y = data_matrix[:, -1]
    return X, Y


def main():
    # 1. Wyznaczenie rankingu cech
    files = glob.glob('sets/*.txt')
    X, Y = get_data(files)

    fvalue_selector = SelectKBest(f_classif)
    fvalue_selector.fit(X, Y) #runung score funtion

    rank = fvalue_selector.scores_
    top_rank = []
    indexes = rank.argsort()[::-1]

    print(len(indexes))

    for index in indexes:
        top_rank.append(rank[index])

    with open('./rank.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file)
        for i, val in enumerate(indexes):
            print('{}: {}'.format(val+1, top_rank[i]))
            writer.writerow([val+1, top_rank[i]])
        

if __name__ == "__main__":
    main()