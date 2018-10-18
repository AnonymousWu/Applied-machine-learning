# SVM
class SVM:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        learning_rate = 1
        epochs = 100000
        lum = 2/epochs

        for epoch in range(epochs):
            for i in range(X.shape[0]):
                if (y[i]*(X[i] @ self.w + self.b)) < 1: 
                    self.w -= learning_rate*(lum*self.w - X[i]*y[i])
                    self.b -= -y[i]
                else:
                    self.w -= learning_rate*(lum/epochs*self.w)


    def predict(self,features):
    # sign( x.w+b )
        prediction = np.sign(features @ self.w +self.b)
        return prediction

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
link = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
dataset = pd.read_csv(link, header=None)
X = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
classifier = SVM();
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))