import numpy as np
import os
import glob
import csv
import pandas

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

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

def classification(X, Y, activation_value, momentum_value, top_rank, layer_size, results):

    # Konfiguracja walidacji krzyzowej - Repeated Stratified K-Fold cross validator z biblioteki scikit-learn
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

    # Konfiguracja modelu - Multi-layer Perceptron z biblioteki scikit-learn
    mlpc = MLPClassifier(hidden_layer_sizes=layer_size,
                        activation=activation_value, max_iter=1000, momentum=momentum_value)

    # Pętla dla walidacji krzyzowej
    for fold, (train, test) in enumerate(rskf.split(X, Y)):
        x_train, x_test = X[train], X[test]
        y_train, y_test = Y[train], Y[test]

        # Dopasowanie x_train do y_train
        mlpc.fit(x_train, y_train)

        # Przypisanie wyniku na podstawie nowych zbiorów
        score = mlpc.score(x_test, y_test)

        # predykcja z pomocą utworzonego klasyfikatora
        # predict = mlpc.predict(x_test)

        # print(score)

        # Tworzenie nowego rekordu z danymi wykonanego badania
        params = {"fold":fold,
            "layer_size":layer_size,
            "momentum_value":momentum_value,
            "activation_value":activation_value,
            "feature_number":len(top_rank)+1,
            "score":score}

        print(params)

        results = results.append(params, ignore_index=True)

    return results


def get_k_best(x, y, k):
    fvalue_selector = SelectKBest(f_classif, k=k)
    fvalue_selector.fit(x, y)
    return fvalue_selector.transform(x)

def main():
    # 1. Wyznaczenie rankingu cech
    files = glob.glob('sets/*.txt')
    X, Y = get_data(files)

    fvalue_selector = SelectKBest(f_classif)
    fvalue_selector.fit(X, Y) #runung score funtion

    rank = fvalue_selector.scores_
    top_rank = []
    indexes = rank.argsort()[::-1]

    print(indexes)

    for index in indexes:
        top_rank.append(rank[index])

    # Funcja do zapisywania pliku z rankinegim cech

    # with open('./rank.csv', mode='w') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for i, val in enumerate(indexes):
    #         print('{}: {}'.format(val+1, top_rank[i]))
    #         writer.writerow([val+1, top_rank[i]])

    # 2. Implementacja środowiska eksperymentowania

    # Liczby neuronów w warstwie ukrytej
    test_layers = [100, 200, 300]

    # Typy funkcji aktywacji
    activation_value = 'relu' # or relu, logistic

    # Wartości momentum
    momentum_values = [0, 0.9] #

    results = pandas.DataFrame(columns=["fold","layer_size","momentum_value", "activation_value", "feature_number","score"])

    filename = "resultaty_relu.csv"

    # Przeprowadzenie badan przy uzyciu roznych konfiguracji zdefiniowanych parametrow
    for layer in test_layers:
            for momentum_value in momentum_values:
                for feature_number in range(1,np.shape(X)[1]+1):

                    print("---------------------------------")
                    print(
                        f"Ilość warstw {layer}, Funkcja aktywacji: {activation_value}, Momentum: {momentum_value}, Liczba cech: {feature_number}")

                    results = classification(X=get_k_best(X, Y, feature_number), Y=Y, activation_value=activation_value, momentum_value=momentum_value, 
                                            top_rank=top_rank[:feature_number - 1], layer_size=layer, results=results)
                                    
    results.to_csv(filename)
        

if __name__ == "__main__":
    main()