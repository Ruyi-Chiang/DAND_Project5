#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi". You will need to use more features
features_list = ['poi',
                 'total_payments', 
                 'total_stock_value', 
                 'from_poi_to_this_person', 
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                 'to_messages', 
                 'deferral_payments', 
                 'expenses', 
                 'deferred_income', 
                 'long_term_incentive', 
                 'restricted_stock_deferred', 
                 'loan_advances', 
                 'from_messages', 
                 'other', 
                 'director_fees', 
                 'bonus', 
                 'restricted_stock', 
                 'salary',  
                 'exercised_stock_options',
                 'fraction_to_poi', # new feature created by myself
                 'fraction_from_poi'] # new feature created by myself 


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### Store to my_dataset for easy export below.
my_dataset = data_dict


### Task 2: Remove outliers
### Remove 2 clear outliers by browsing datapoints' names
my_dataset.pop('THE TRAVEL AGENCY IN THE PARK', 0)
my_dataset.pop('TOTAL', 0)


def outlier_identifier(dataset, feature): # Using Tukey's Inter Quartile Range method
    values_list = []
    for _, v in dataset.iteritems():
        if v[feature] != 'NaN':
            values_list.append(v[feature])
    import numpy as np
    
    Q1 = np.percentile(values_list, q = 25)
    Q3 = np.percentile(values_list, q = 75)
    iqr = Q3 - Q1
    lower_fence = Q1 - 1.5*iqr
    upper_fence = Q3 + 1.5*iqr
    
    outlier_cnt = 0
    outlier_values = []
    for value in values_list:
        if value > upper_fence or value < lower_fence:
            outlier_cnt +=1
            outlier_values.append(value)
    return outlier_cnt, outlier_values

def mapping_to_name(dataset, feature, value):
    for k, v in dataset.iteritems():
        if v[feature] == value:
            return k

# a function to return names of outliers and show if they are poi or not
# if the outliers belong to poi, I would keep them in the dataset. Otherwise, I would remove outliers by name.
def outlier_detector(dataset, feature):
    cnt, outlier_values = outlier_identifier(dataset, feature)
    name_list = []
    for value in outlier_values:
        name_list.append(mapping_to_name(dataset, feature, value))
    poi_values = {}
    for name in name_list:
        poi_values[name] = dataset[name]['poi']
    poi_name_outlier = []
    non_poi_name_outlier = []
    for k, v in poi_values.iteritems():
        if v == True:
            poi_name_outlier.append(k)
        else:
            non_poi_name_outlier.append(k)
    result = {'poi_outliers': poi_name_outlier, 'non_poi_outliers': non_poi_name_outlier}
    return result

def remove_outliers(dataset, feature):
    outlier_key_list = outlier_detector(dataset, feature)
    outlier_key_list = outlier_key_list['non_poi_outliers']
    for key in outlier_key_list:
        dataset.pop(key, 0)

# Checking and removing outliers in financial features: 'total_payments', 'total_stock_value'
remove_outliers(my_dataset, 'total_payments')


### Task 3: Create new feature(s)


##  Create 2 new features called fraction of emails from/to poi
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    if poi_messages != "NaN" or all_messages != "NaN":
        fraction = float(poi_messages) / float(all_messages)
    else:
        fraction = 0
    return fraction


## Added 2 new features into dataset - fraction of emails from/to poi
for name in my_dataset:
    data_point = my_dataset[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi

all_features_list = []
for _, v in my_dataset.iteritems():
    for k2, _ in v.iteritems():
        if k2 not in all_features_list:
            all_features_list.append(k2)
print "Total number of features:", len(all_features_list) # Number of all features should be 23 by now.

# Checking and Removing Email features: 'fraction_from_poi', 'fraction_to_po'
remove_outliers(my_dataset, 'fraction_from_poi')
remove_outliers(my_dataset, 'fraction_to_poi')
# I found that removing email outliers a few times help to improve my algorithm result
remove_outliers(my_dataset, 'fraction_from_poi')
remove_outliers(my_dataset, 'fraction_to_poi')
remove_outliers(my_dataset, 'fraction_from_poi')
remove_outliers(my_dataset, 'fraction_to_poi')


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.4, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from clfs import Kneightbors_clf, LinearSVC_clf, naive_bayes_clf, SVC_clf
Kneightbors_clf(features_train = features_train, \
                labels_train = labels_train,\
                features_test =features_test, \
                labels_test = labels_test)
LinearSVC_clf(features_train = features_train, \
                labels_train = labels_train,\
                features_test =features_test, \
                labels_test = labels_test)
SVC_clf(features_train = features_train, \
                labels_train = labels_train,\
                features_test =features_test, \
                labels_test = labels_test)

naive_bayes_clf(features_train = features_train, \
                labels_train = labels_train,\
                features_test =features_test, \
                labels_test = labels_test)

print 'SVC classifier returns the highest accuracy score\
 so I am gonna tune this classifier with GridSearchCV'


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!



from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
#from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, svm
from sklearn.grid_search import GridSearchCV #sklearn version 0.15
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


scaler = preprocessing.MinMaxScaler()
skb = SelectKBest()

clf = svm.SVC()
pipe = Pipeline([('Scale_Features',scaler),('SKB', skb),('Classifier',clf)])

# print sorted(pipe.get_params().keys())

params = {'SKB__k':range(1,len(features_list)),'Classifier__kernel':['linear', 'rbf'], 'Classifier__C':[1, 10]}
my_clf = GridSearchCV(pipe, param_grid=params, scoring='f1_weighted')
my_clf.fit(features_train, labels_train)


pred = my_clf.predict(features_test)
print("Best estimator found by grid search:")
print my_clf.best_estimator_
print('Best Params found by grid search:')
print my_clf.best_params_

print('Selected Features:')
features_selected_bool = my_clf.best_estimator_.named_steps['SKB'].get_support()
features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]

print features_selected_list, '\n'

print ('My fine-tuned Algorithm')
print('Best Score found by grid search:')  
print my_clf.best_score_      
print 'Accuracy:',accuracy_score(labels_test,pred)    
print 'Precision:',precision_score(labels_test,pred)    
print 'Recall:',recall_score(labels_test,pred)    
print 'F1 Score:',f1_score(labels_test,pred)

clf = my_clf.best_estimator_







print ('Using the from_this_person_to_poi')
test_eatures_list = ['poi','from_this_person_to_poi']
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, test_eatures_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train_original, features_test_original, labels_train_original, labels_test_original = \
    train_test_split(features, labels, test_size=0.4, random_state=42)

second_clf = svm.SVC(kernel ='linear', C =10)
second_clf.fit(features_train_original, labels_train_original)
pred = second_clf.predict(features_test_original)
print 'Accuracy:',accuracy_score(labels_test_original,pred)    
print 'Precision:',precision_score(labels_test_original,pred)    
print 'Recall:',recall_score(labels_test_original,pred)    
print 'F1 Score:',f1_score(labels_test_original,pred)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)