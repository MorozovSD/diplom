import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_graph(data, title='', x_label='', y_label=''):
    ts = pd.Series(data)
    ts.plot()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
