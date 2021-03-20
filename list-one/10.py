import math
import random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def calc_distance(x, y):
    sum = 0
    for i in range(4):
        sum += math.pow(float(x[i]) - float(y[i]), 2)
    return math.sqrt(sum)


def knn(training_set, test_el, K):
    dists = {}
    for i in range(len(training_set)):
        d = calc_distance(training_set[i], test_el)
        dists[i] = d

    kn = sorted(dists, key=dists.get)[:K]

    total_setosa, total_versicolor, total_virginica = 0, 0, 0

    for i in kn:
        if training_set[i][-1] == 'Iris-setosa':
            total_setosa += 1
        elif training_set[i][-1] == 'Iris-versicolor':
            total_versicolor += 1
        else:
            total_virginica += 1
    a = [total_setosa, total_versicolor, total_virginica]

    if a.index(max(a)) == 0:
        cl = 'Iris-setosa'
    elif a.index(max(a)) == 1:
        cl = 'Iris-versicolor'
    else:
        cl = 'Iris-virginica'

    return cl


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


data, k_scores, pred, test = [], [], [], []
k_range = range(1, 31)

with open('iris.data', 'r') as f:
    for line in f.readlines():
        a = line.replace('\n', '').split(',')
        data.append(a)

random.shuffle(data)
splitted_data = list(chunks(data, 30))

for k in k_range:
    pred = []
    test = []

    for i in range(len(splitted_data)):
        test_set = splitted_data[i]

        splitted_data.remove(test_set)

        training_set = [j for l in splitted_data for j in l]

        for el in test_set:
            group = knn(training_set, el, k)

            pred.append(group)
            test.append(el[-1])

        splitted_data.append(test_set)

    f1 = f1_score(test, pred, average="micro")
    k_scores.append(f1)
    print(f"k: {k} - F1-score: {f1}")

plt.plot(k_range, k_scores)

plt.xlabel('Valor do k no KNN')
plt.ylabel('F1-score')
plt.show()
