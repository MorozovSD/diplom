from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from random import randint
from sklearn.tree import DecisionTreeClassifier
from map_inf import map_inf


def get_random_samples(data, labels, length):
    X = []
    y = []
    datalength = len(data) - 1

    for i in range(length):
        X.append(data[randint(0, datalength)])
        y.append(labels[randint(0, datalength)])

    return X, y


def get_data(length, categories=None, binary=False):
    dataset = datasets.fetch_20newsgroups(shuffle=False, categories=categories)
    data = dataset.data
    labels = dataset.target
    data, labels = get_random_samples(data, labels, length)

    vect = CountVectorizer(binary=binary)
    matrix = vect.fit_transform(data)
    # features = vect.get_feature_names()

    X = matrix.toarray()
    y = labels

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    k = len(set(y))

    return X, y, map_inf(clf, X, k)
