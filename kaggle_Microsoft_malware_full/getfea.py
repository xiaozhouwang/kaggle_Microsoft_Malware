from sklearn.ensemble import RandomForestClassifier
import pickle
import sys
import numpy as np

X1=np.array(pickle.load(open('X2g_train.p')))
X2=np.array(pickle.load(open('X3g_train.p')))
X3=np.array(pickle.load(open('X4g_train.p')))
X4=np.array(pickle.load(open('Xhead_train.p')))

X=np.hstack((X2,X1,X3,X4))
y=np.array(pickle.load(open('y.p')))
rf=RandomForestClassifier(n_estimators=200)
Xr=rf.fit_transform(X,y)
pickle.dump(Xr,open('X33_train_reproduce.p','w'))
print Xr.shape
del X,X1,X2,X3,X4,Xr

X1=np.array(pickle.load(open('X2g_test.p')))
X2=np.array(pickle.load(open('X3g_test.p')))
X3=np.array(pickle.load(open('X4g_test.p')))
X4=np.array(pickle.load(open('Xhead_test.p')))
X=np.hstack((X2,X1,X3,X4))
Xr=rf.transform(X)
pickle.dump(Xr,open('X33_test_reproduce.p','w'))
print Xr.shape
