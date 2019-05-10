from data import get_data
from dom_inf import dom_inf
from cod_inf import cod_inf

X, y, l = get_data(100, categories=['sci.space', 'sci.crypt'], binary=True)
dom_inf1 = dom_inf(X)
cod_inf1 = cod_inf(y)
print("Sample 1")
print("{:.2f}".format(dom_inf1))
print("{:.2f}".format(cod_inf1))
print("{:.2f}".format(l))

X, y, l = get_data(100)
dom_inf2 = dom_inf(X)
cod_inf2 = cod_inf(y)
print("Sample 2")
print("{:.2f}".format(dom_inf2))
print("{:.2f}".format(cod_inf2))
print("{:.2f}".format(l))

X, y, l = get_data(10)
dom_inf3 = dom_inf(X)
cod_inf3 = cod_inf(y)
print("Sample 3")
print("{:.2f}".format(dom_inf3))
print("{:.2f}".format(cod_inf3))
print("{:.2f}".format(l))

print("Comparing")
print("{:.2f}".format(dom_inf1 / dom_inf2))
print("{:.2f}".format(cod_inf1 / cod_inf2))
print("{:.2f}".format(cod_inf2 / cod_inf3))
print("{:.2f}".format(cod_inf1 / cod_inf3))
