# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:38:13 2021

@author: joelr
"""

import numpy
import matplotlib.pyplot as plt

numpy.random.seed(1)
x = numpy.random.normal(3, 1, 500)
y = numpy.random.normal(50, 20, 500) / x

plt.scatter(x, y)
plt.show()

train_x = x[:400]
train_y = y[:400]

test_x = x[400:]
test_y = y[400:]

plt.scatter(train_x, train_y)
plt.show()

plt.scatter(test_x, test_y)
plt.show()

#Draw a polynomial regression line through the data points:
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()


# The result can back my suggestion of the data set fitting a polynomial regression, even though it would give us some weird results if we try to predict values outside of the data set. Example: the line indicates that a customer spending 6 minutes in the shop would make a purchase worth 200. That is probably a sign of overfitting.

# But what about the R-squared score? The R-squared score is a good indicator of how well my data set is fitting the model.

#How well does my training data fit in a polynomial regression? r2=0 pouca correlação e r2=1 muita correlação
from sklearn.metrics import r2_score

# Testando conjunto de treino 
r2 = r2_score(train_y, mymodel(train_x))
print(r2)
# Testando conjunto de treino 
r2 = r2_score(test_y, mymodel(test_x))
print(r2)

#Predict Values - How much money will a buying customer spend, if she or he stays in the shop for 5 minutes?

print(mymodel(1))


