from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

olivetti_faces=fetch_olivetti_faces()
features=olivetti_faces.data
targets=olivetti_faces.target

x_train,x_test,y_train,y_test=train_test_split(features,targets,test_size=0.3,stratify=targets)
pca=PCA(n_components=200,whiten=True)
pca.fit(x_train)
X_train_pca=pca.transform(x_train)
X_test_pca=pca.transform(x_test)

models = [('Logistic Regression',LogisticRegression()),('Naive Bayes',GaussianNB()),('Support Vector Machine',SVC())]
for name,model in models:
    classifier_model=model
    classifier_model.fit(X_train_pca,y_train)
    y_predicted=classifier_model.predict(X_test_pca)
    print("Results with %s "%name)
    print("Accuracy score %s "%metrics.accuracy_score(y_test,y_predicted))
