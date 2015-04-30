import sys
import pickle

##########################################################
# usage
# pypy findhead.py xid_train.p ../../data/train 

# xid_train.p is a list like ['loIP1tiwELF9YNZQjSUO',''....] to specify
# the order of samples in traing data
# ../../data/train is the path of original train data
##########################################################
xid_name=sys.argv[1]
data_path=sys.argv[2]


xid=pickle.load(open(xid_name)) #xid_train.p or xid_test.p

head={}

for c,f in enumerate(xid):
    fo=open(data_path+'/'+f+'.asm')
    tot=0
    for line in fo:
        xx=line.split()
        h=xx[0].split(':')
        if h[0] not in head:
            head[h[0]]=0
        head[h[0]]+=1 
    fo.close()
    if True:#c%10000==0:
        print c*1.0/len(xid),len(head)
print len(head)
pickle.dump(head,open('head.p','w'))
