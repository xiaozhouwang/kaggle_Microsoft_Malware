from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LGR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import ExtraTreesClassifier as ET
from xgboost_multi import XGBC
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
import pickle

# create model_list

def get_model_list():
    model_list = []
    for num_round in [200]:
        for max_depth in [1]:
            for eta in [0.25]:
                for min_child_weight in [1]:
                    for col_sample in [0.8]:
                        model_list.append((XGBC(num_round = num_round, max_depth = max_depth, eta = eta, 
                                                min_child_weight = min_child_weight, colsample_bytree = col_sample),
                                           'xgb_tree_%i_depth_%i_lr_%f_child_%i_col_sample_%i'%(num_round, max_depth, eta, min_child_weight,col_sample)))


    return model_list



def gen_data():

    # the 4k features!
    the_train = pickle.load(open('X33_train_reproduce.p','rb'))  
    the_test = pickle.load(open('X33_test_reproduce.p','rb'))
    # corresponding id and labels
    Id = pickle.load(open('xid.p','rb'))
    labels = pickle.load(open('y.p','rb'))    
    Id_test = pickle.load(open('Xt_id.p','rb'))

    # merge them into pandas
    join_train = np.column_stack((Id, the_train, labels))
    join_test = np.column_stack((Id_test, the_test))
    train = pd.DataFrame(join_train, columns=['Id']+['the_fea%i'%i for i in xrange(the_train.shape[1])] + ['Class'])
    test = pd.DataFrame(join_test, columns=['Id']+['the_fea%i'%i for i in xrange(the_train.shape[1])])
    del join_train, join_test
    # convert into numeric features
    train = train.convert_objects(convert_numeric=True)
    test = test.convert_objects(convert_numeric=True)
    
    # including more things
    train_count = pd.read_csv("train_frequency.csv")
    test_count = pd.read_csv("test_frequency.csv") 
    train = pd.merge(train, train_count, on='Id')
    test = pd.merge(test, test_count, on='Id')


    
    # instr count
    train_instr_count = pd.read_csv("train_instr_frequency.csv")
    test_instr_count = pd.read_csv("test_instr_frequency.csv")
    for n in list(train_instr_count)[1:]:
        if np.sum(train_instr_count[n]) == 0:
            del train_instr_count[n]
            del test_instr_count[n]
    
    train_instr_freq = train_instr_count.copy()
    test_instr_freq = test_instr_count.copy()
    
    train_instr_freq.ix[:,1:] = train_instr_freq.ix[:,1:].apply(lambda x: x/np.sum(x), axis = 1)
    #train_instr_freq = train_instr_freq.replace(np.inf, 0)
    train_instr_freq = train_instr_freq.replace(np.nan, 0)
    test_instr_freq.ix[:,1:]=test_instr_freq.ix[:,1:].apply(lambda x: x/np.sum(x), axis = 1)
    #test_instr_freq = test_instr_freq.replace(np.inf, 0)
    test_instr_freq = test_instr_freq.replace(np.nan, 0)
    
    train = pd.merge(train, train_instr_freq, on='Id')
    test = pd.merge(test, test_instr_freq, on='Id')    
    
    ## all right, include more!
    grams_train = pd.read_csv("train_data_750.csv")
    grams_test = pd.read_csv("test_data_750.csv")
    
    # daf features
    #train_daf = pd.read_csv("train_daf.csv")
    #test_daf = pd.read_csv("test_daf.csv")
    #daf_list = [0,165,91,60,108,84,42,93,152,100] #daf list for 500 grams.
    # dll features
    train_dll = pd.read_csv("train_dll.csv")
    test_dll = pd.read_csv("test_dll.csv")

    
    # merge all them
    #mine = pd.merge(grams_train, train_daf,on='Id')
    mine = grams_train
    mine = pd.merge(mine, train_dll, on='Id')
    
    mine_labels = pd.read_csv("trainLabels.csv")
    mine = pd.merge(mine, mine_labels, on='Id')
    mine_labels = mine.Class
    mine_Id = mine.Id
    del mine['Class']
    del mine['Id']
    mine = mine.as_matrix()

    #mine_test = pd.merge(grams_test, test_daf,on='Id')
    mine_test = grams_test
    mine_test = pd.merge(mine_test, test_dll,on='Id')

    mine_test_id = mine_test.Id
    del mine_test['Id']
    clf_se = RF(n_estimators=500, n_jobs=-1,random_state = 0)
    clf_se.fit(mine,mine_labels)
    mine_train = np.array(clf_se.transform(mine, '1.25*mean'))
    mine_test = np.array(clf_se.transform(mine_test, '1.25*mean'))

    train_mine = pd.DataFrame(np.column_stack((mine_Id, mine_train)), columns=['Id']+['mine_'+str(x) for x in xrange(mine_train.shape[1])]).convert_objects(convert_numeric=True)
    test_mine = pd.DataFrame(np.column_stack((mine_test_id, mine_test)), columns=['Id']+['mine_'+str(x) for x in xrange(mine_test.shape[1])]).convert_objects(convert_numeric=True)
    train = pd.merge(train, train_mine, on='Id')
    test = pd.merge(test, test_mine, on='Id')

    train_image = pd.read_csv("train_asm_image.csv", usecols=['Id']+['asm_%i'%i for i in xrange(800)])
    test_image = pd.read_csv("test_asm_image.csv", usecols=['Id']+['asm_%i'%i for i in xrange(800)])
    train = pd.merge(train, train_image, on='Id')
    test = pd.merge(test, test_image, on='Id')
    print "the data dimension:"
    print train.shape, test.shape
    return train, test
def gen_submission(model):
    # read in data
    print "read data and prepare modelling..."
    train, test = gen_data()
    X = train
    Id = X.Id
    labels = np.array(X.Class - 1) # for the purpose of using multilogloss fun.
    del X['Id']
    del X['Class']
    X = X.as_matrix()
    X_test = test
    id_test = X_test.Id
    del X_test['Id']
    X_test = X_test.as_matrix()   

    clf, clf_name = model
    print "generating model %s..."%clf_name
    clf.fit(X, labels)
    pred = clf.predict_proba(X_test)
    pred = pred.reshape(X_test.shape[0],9)
    pred = np.column_stack((id_test, pred))
    submission = pd.DataFrame(pred, columns=['Id']+['Prediction%i'%i for i in xrange(1,10)])
    submission = submission.convert_objects(convert_numeric=True)    
    submission.to_csv('model3.csv',index = False)


def cross_validate(model_list):

    # read in data
    print "read data and prepare modelling..."
    train, test = gen_data()
    X = train
    Id = X.Id
    labels = np.array(X.Class - 1) # for the purpose of using multilogloss fun.
    del X['Id']
    del X['Class']
    X = X.as_matrix()
    X_test = test
    id_test = X_test.Id
    del X_test['Id']
    X_test = X_test.as_matrix()

    kf = KFold(labels, n_folds=4)  # 4 folds

    best_score = 1.0
    for j, (clf, clf_name) in enumerate(model_list):
        print "modelling %s"%clf_name
        stack_train = np.zeros((len(Id),9)) # 9 classes.
        for i, (train_fold, validate) in enumerate(kf):
            X_train, X_validate, labels_train, labels_validate = X[train_fold,:], X[validate,:], labels[train_fold], labels[validate]          
            clf.fit(X_train,labels_train)
            stack_train[validate] = clf.predict_proba(X_validate)
        print multiclass_log_loss(labels, stack_train)
        if multiclass_log_loss(labels, stack_train) < best_score:
            best_score = multiclass_log_loss(labels, stack_train)
            best_selection = j

    return model_list[best_selection]


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss


if __name__ == '__main__':
    model_list = get_model_list()
    #best_model = cross_validate(model_list)
    gen_submission(model_list[0])#0.0051
    print "ALL DONE!!!"


