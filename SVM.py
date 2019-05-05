from numpy import genfromtxt
from sklearn import svm
from sklearn.svm import SVC
import numpy as np

trainData = genfromtxt("train_samples.csv", delimiter = ",")
testData = genfromtxt("test_samples.csv", delimiter = ",")
trainLabels = genfromtxt("train_labels.csv", delimiter = ",")
trainLabels = trainLabels.astype(int)

def confusion_matrix(labels_true, labels_predicted):
    cm = np.zeros((8,8), 'int')

    for i in range(len(labels_true)):
        cm[labels_true[i], labels_predicted[i]] = cm[labels_true[i], labels_predicted[i]] + 1 #c[i][j] = should be label i and was predicted label j

    return cm

from sklearn.model_selection import KFold #class used for spliting the trainData
kf = KFold(n_splits = 3)
kf.get_n_splits(trainData)

nr_fold = 0
sumAccuracy = 0

for trainIndex, testIndex in kf.split(trainData):
    nr_fold = nr_fold + 1 #current fold

    #trainData and trainLabels for fold nr_fold_
    trainData3Cross = trainData[trainIndex]
    trainLabels3Cross = trainLabels[trainIndex]

    #testData and testLabels for fol nr_fold
    testData3Cross = trainData[testIndex]
    testLabels3Cross = trainLabels[testIndex]
    testLabels3Cross = testLabels3Cross.astype(int)

    #training the svm classifier on each fold
    cf = svm.SVC(gamma = 'scale')
    cf.fit(trainData3Cross, trainLabels3Cross)
    predictions = cf.predict(testData3Cross)
    predictions = predictions.astype(int) #the labels predicted by the classifier


    accuracy = (predictions == testLabels3Cross).mean()
    sumAccuracy = sumAccuracy + accuracy
    print("Accuracy for fold ", nr_fold, ": ", accuracy)
    print("Confusion matrix fold ", nr_fold, ": ")
    print(confusion_matrix(testLabels3Cross, predictions))

print("Mean accuracy: ", sumAccuracy/3)


#training the svm classifier on all the trainData and prediction made on the real testData
cf = svm.SVC(gamma='scale')
cf.fit(trainData, trainLabels)
predictions = cf.predict(testData)


#write the predictions in a csv file
import csv
with open('TLabels.csv', 'w', newline = '') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Id', 'Prediction'])

    for i in range(5000):
        filewriter.writerow([i+1, predictions[i]])