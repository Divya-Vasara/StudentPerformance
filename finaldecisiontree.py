import time
import graphviz
import pandas as pd
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sklearn.metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
start_time = time.time()


df = pd.read_csv(r'data.csv')
df['sex'] = df['sex'].map({'M': 0, 'F': 1})
#df['address'] = df['address'].map({'U': 0, 'R': 1})
df['guardian'] = df['guardian'].map({'mother': 0, 'father': 1})
#df['internet'] = df['internet'].map({'no': 0, 'yes': 1})


predictors =np.nan_to_num(df.values[:, 0:14]) 
print(predictors[0:5])
targets = df.values[:,14]
print(targets[0:5])
#predictors.fillna(predictors.mean(), inplace=True)
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size= 0.25)


print(pred_train.shape)
print(pred_test.shape)
print(tar_train.shape)
print(tar_test.shape)

features = list(df.columns[:11])

classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 1, splitter='best')
#lassifier = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto')

'''classifier=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')
#max_depth=6, min_samples_leaf=7
#bagging = BaggingClassifier(DecisionTreeClassifier())'''
classifier = classifier.fit(pred_train,tar_train)
print("dddddddddddddddddddddddddddddd")
print(pred_test[0:5])
predictions = classifier.predict(pred_test)
#print("dddddddddddddddddddddddddddddd")
#print(predictions[0:5])

print(sklearn.metrics.confusion_matrix(tar_test, predictions))

#classification accuracy
print("accuracy of training dataset is{:.2f}".format(classifier.score(pred_train,tar_train)))
print("accuracy of test dataset is {:.2f}".format(classifier.score(pred_test,tar_test)))
#print(accuracy_score(tar_test, predictions, normalize = True))

#error rate
print("Error rate is",1- accuracy_score(tar_test, predictions, normalize = True))

#sensitivity
print("sensitivity is", sklearn.metrics.recall_score(tar_test, predictions,labels=None, average =  'micro', sample_weight=None))
#specificity
print("specificity is", 1 - sklearn.metrics.recall_score(tar_test, predictions,labels=None, average =  'micro', sample_weight=None))

#precision
#print("precision is",sklearn.metrics.precision_score(tar_test, predictions, labels=None, pos_label=1, average =  'micro' ,sample_weight=None))

#Recall


from sklearn import metrics
from sklearn import model_selection
models = []
#models.append(('Random Forest Classifier', RFC_Classifier))
models.append(('Decision Tree Classifier', classifier))
#models.append(('Support Vector Classifier',SVM_Classifier))
import numpy as np
from sklearn.model_selection import validation_curve 
import seaborn as sns
results = []
seed = 7
names = []
scoring = 'accuracy'
for i, v in models:
    Xpred =  v.predict(pred_train)
    print("predict")
    #print(Xpred[0:10])
    scores = cross_val_score(v, pred_train, tar_train, cv=10)
    results.append(scores)
    names.append(v)
    accuracy = metrics.accuracy_score(tar_train, Xpred)
    confusion_matrix = metrics.confusion_matrix(tar_train, Xpred)
    classification = metrics.classification_report(tar_train, Xpred)
    msg = "%s: %f (%f)" % (name, results.mean(), results.std())
	print(msg)
    print()
    print('============================== {} Model Evaluation =============================='.format(i))
    print()
    #print ("Cross Validation Mean Score:" "\n", scores.mean())
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    #print("Confusion matrix:" "\n", confusion_matrix)
    #print()
    #print("Classification report:" "\n", classification) 
    #print()
    plt.figure(figsize=(5, 7))
    ax = sns.distplot(targets, hist=False, color="r", label="Actual Value")
    sns.distplot(Xpred, hist=False, color="b", label="predicted Values" , ax=ax)
    plt.title('Actual vs predicted Values ')
    plt.show()
    plt.close() 

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

 





