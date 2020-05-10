# python 3.6

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn import tree

data = pd.read_csv("data/train.csv").to_numpy()
clf = tree.DecisionTreeClassifier()

xtrain = data[0:21000, 1:]
xtest = data[21000:, 1:]
train_label = data[0:21000, 0]

clf.fit(xtrain, train_label)

def testmachine(number):
     d = xtest[number]
     d.shape = (28, 28)
     plot.imshow(255 - d, cmap="gray")
     print(clf.predict([xtest[number]]))
     plot.show()

num = int(input("Number: "))
testmachine(num)

def accuracy():
     actual_label = data[21000:,0]
     predict = clf.predict(xtest)

     n = 0
     for i in range(0, 21000):
          if predict[i] == actual_label[i]:
               n += 1
     print("Accuracy:", (n/21000)*100)

accuracy()