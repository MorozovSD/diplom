import random

from math import log

from graph_writer import to_graph


class Iris:
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width, iris_type):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.type = iris_type

    def to_dict(self):
        return {'sepal_length': self.sepal_length,
                'sepal_width': self.sepal_width,
                'petal_length': self.petal_length,
                'petal_width': self.petal_width,
                'type': self.type}


def load_data():
    iris_list = []
    with open('./iris_data') as file:
        file_data = file.read().split('\n')
        for line in file_data:
            sepal_length, sepal_width, petal_length, petal_width, iris_type = line.split('\t')
            iris_list.append(Iris(sepal_length=sepal_length,
                                  sepal_width=sepal_width,
                                  petal_length=petal_length,
                                  petal_width=petal_width,
                                  iris_type=iris_type))
    return iris_list


# Информативность области определения
# d(yi) = x1...xnyi∈Yi.
# l(d(Xj) = log|Xj|
# l(d(yi) = ∑l(d(Xj))
# K(yi) = l(d(yi))

# Информативность области значений
# P(yi) = |Yi| / |T|,
# I(yi) = −log|Yi| / |T|


if __name__ == "__main__":

    # T = {x1...xny : xi ∈ Xi, y ∈ Y },
    iris_data = load_data()
    random.shuffle(iris_data)
    dom_inf_dict = {}
    cod_inf_dict = {}

    for i in range(2, len(iris_data)):
        test_iris_data = iris_data[:i]
        x1_capacity = len(set(_iris.sepal_length for _iris in test_iris_data))
        x2_capacity = len(set(_iris.sepal_width for _iris in test_iris_data))
        x3_capacity = len(set(_iris.petal_length for _iris in test_iris_data))
        x4_capacity = len(set(_iris.petal_width for _iris in test_iris_data))

        n = x1_capacity + x2_capacity + x3_capacity + x4_capacity
        inf = [log(x, n) for x in [x1_capacity, x2_capacity, x3_capacity, x4_capacity]]
        # изменить mean (смотри нампай)
        mean_inf = sum(inf)/len(inf)
        dom_inf_dict[i] = mean_inf

    to_graph(dom_inf_dict,
             y_label='Информативность области определения',
             x_label='Количество входных данных')
    print('Информативность области определения')
    for key in dom_inf_dict.keys():
        print(f'{key}\t{dom_inf_dict[key]}')

    # Yi = {x1...xny: y = yi}.
    iris_types_dict = {}
    for i in range(2, len(iris_data)):
        test_iris_data = iris_data[:i]
        for iris in test_iris_data:
            if iris_types_dict.get(iris.type):
                iris_types_dict[iris.type].append(iris.to_dict())
            else:
                iris_types_dict[iris.type] = [(iris.to_dict())]
        y_capacity = []
        for iris_type in iris_types_dict.keys():
            y_capacity.append(len(iris_types_dict[iris_type]))
        n = sum(y_capacity)
        inf = [log(x, n) for x in y_capacity]

        # изменить mean (смотри нампай)
        mean_inf = sum(inf)/len(inf)
        cod_inf_dict[i] = mean_inf

    to_graph(cod_inf_dict,
             y_label='Информативность области значения',
             x_label='Количество входных данных')
    print('Информативность области значения')
    for key in cod_inf_dict.keys():
        print(f'{key}\t{cod_inf_dict[key]}')

    # область определения X = X1 × ... × Xn;
    # область определения X состоит из параметров X1, .., Xn;
    # для параметра Xi количество значений |Xi|;

    # область значений Y = Y1 ∪ ... ∪ Yk.
    # range_of_values = iris_types_dict.keys()

    # область значений Y состоит из подмножеств Y1, ..., Yk
    # subsets_of_values = iris_types_dict

    # для подмножества значений Yi : объем |Yi|.
    # capacity_of_subsets_of_values = {}
    # for i, iris_type in enumerate(iris_types_dict.keys()):
    #     i += 1
    #     capasity_of_subsets_of_values[f'Y{i}'] = len(iris_types_dict[iris_type])
    # print(capasity_of_subsets_of_values)

    # множество отображений F = Y^X;
    # множество отображений F состоит из подмножеств композиций F1, ..., Fm определенной длины;
    # для подмножества отображений Fi : объем |Fi|;

    # for iris_type in iris_types_dict.keys():
    #     print(iris_type)
    #     for iris_parameters in iris_types_dict[iris_type]:
    #         print(f'\t {iris_parameters}')
