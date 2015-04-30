import os
xid=[i.split('.')[0] for i in os.listdir('train') if '.asm' in i]
Xt_id=[i.split('.')[0] for i in os.listdir('test') if '.asm' in i]
f=open('trainLabels.csv')
f.readline()
label={}
for line in f:
    xx=line.split(',')
    idx=xx[0][1:-1]
    label[idx]=int(xx[-1])
f.close()
y=[label[i] for i in xid]
import pickle
pickle.dump(xid,open('xid_train.p','w'))
pickle.dump(Xt_id,open('xid_test.p','w'))
pickle.dump(xid,open('xid.p','w'))
pickle.dump(Xt_id,open('Xt_id.p','w'))
pickle.dump(y,open('y.p','w'))
