'''
    membuat model untuk proses klasifikasi menggunkan SVM 
    inspirasi dari https://github.com/sbhusal123/Speaker-Recognition/
    dimana dari repository menggunakan SVM
    disini mencoba menggunakan roadmap dari repository yaitu :
    
    1. First Phase:
        a. Just focus on developing the model.
        b. Weather it be full of accuracy or not. Dont't think of accuracy at first
        c. Use SVM with any parameters.
    2. Second Phase
        a. Try to optimize the model's hyper-parameter
        b. Focus only on hyper-parameters
    3. Third Phase:
        a. Try to build the functionality of data pre-processing
        b. Add extra functionality
'''

'''
    tahap sekarang : first phase
'''

import pickle
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split,GridSearchCV
from FeatureExtraction import ExtractFeature
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
'''
    dokumentasi SVC ada di https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

    entah kenapa ini ada masalah waktu buat modelnya, misalnya
    sekarang kita buat trs dpt akurasi diangka 90% bsk nya malah turun signifikan atau naik

    detail akurasi terkini beserta konfigurasi

    kernel Linear (C = 1.0) => 100% 
    kernel RBF (parameter default) => 86% 
    kernel Poly (C = 100.0, degree =1) => 100%
    kernel Sigmoid (C=100.0) => 72%

'''
def make_model(dataset_location):
    dataset = pd.read_csv(dataset_location)
    X = dataset.drop('speaker',axis=1)
    Y = dataset['speaker']
    
    X = X.values
    X_test = X
 
    #PCA SCALLING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA().fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.2)
    
    svc_classifier = any
    # if(kernel == 'linear'):
    #     svc_classifier = SVC(kernel=kernel, C = 4, gamma = 0.01,probability=True)
    # elif(kernel == 'rbf'):
    #     # accuracy 98 % using configuration => random_state=0, gamma=.01, C=1
    #     # edit 1 : itu parameter diatas entah kenapa baru tak coba malah akurasinya 0 porsen
    #     # edit 2 : kalau tanpa parameter akurasinya 86 %
    #     svc_classifier = SVC(kernel=kernel,C = 10, gamma = 0.01,probability=True)
    # elif(kernel == 'poly'):
    #     # accuracy 100% using configuration C=100.0, degree=1
    #     svc_classifier = SVC(kernel=kernel, C=100.0, degree=1)
    # elif(kernel == 'sigmoid'):
    #     svc_classifier = SVC(kernel=kernel,C=100.0)
    # else:
    #     kernel = 'linear_svc'
    #     svc_classifier = LinearSVC(C=1.0).fit(X_train, Y_train)
    param_grid = [
        {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'degree':[1,2,3,4,5,6,7,8,9,10], 'kernel': ['poly']},
    ]
    svc = SVC()
    svc_classifier= GridSearchCV(svc, param_grid)
    svc_classifier.fit(X_train,Y_train)
    y_pred = svc_classifier.predict(X_test)
    heat_map = sns.heatmap(confusion_matrix(Y_test,y_pred))
    print(classification_report(Y_test,y_pred))
    print(svc_classifier.best_estimator_)

    cr = classification_report(Y_test,y_pred)
    svc = svc_classifier.best_estimator_

    plt.show()
    pickle.dump(svc_classifier,open("dataset/model"+".pickle",'wb'))

    return cr, svc

#make_model("dataset/dataset_train.csv")