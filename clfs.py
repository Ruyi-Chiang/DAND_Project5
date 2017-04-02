import time

### KNeighbors clf
def Kneightbors_clf(features_train,labels_train, \
                    features_test,labels_test):

    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors = 2)
    
    print 'clf name: KNeighbors'
    t0 = time.time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time.time()-t0, 3), "s"

    t0 = time.time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time.time()-t0, 3), "s"
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)
    print 'accuracy score:', accuracy


### SVC clf

def SVC_clf(features_train,labels_train, \
                    features_test,labels_test):

    from sklearn import svm
    clf = svm.SVC()
    
    print 'clf name: SVC'
    t0 = time.time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time.time()-t0, 3), "s"

    t0 = time.time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time.time()-t0, 3), "s"

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)
    print 'accuracy score:', accuracy


### LinearSVC clf

def LinearSVC_clf(features_train,labels_train, \
                    features_test,labels_test):

    from sklearn import svm
    clf = svm.LinearSVC()
    
    print 'clf name: LinearSVC'
    t0 = time.time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time.time()-t0, 3), "s"

    t0 = time.time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time.time()-t0, 3), "s"

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)
    print 'accuracy score:', accuracy

def naive_bayes_clf(features_train,labels_train, \
                    features_test,labels_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    
    print 'clf name: Naive Bayes'
    t0 = time.time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time.time()-t0, 3), "s"

    t0 = time.time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time.time()-t0, 3), "s"

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)
    print 'accuracy score:', accuracy