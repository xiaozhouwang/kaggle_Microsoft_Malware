import sys
import pickle

##########################################################
# usage
# pypy rebuild_2g.py xid_train.p ../../data/train 

# xid_train.p is a list like ['loIP1tiwELF9YNZQjSUO',''....] to specify
# the order of samples in traing data
# ../../data/train is the path of original train data
##########################################################

xid_name=sys.argv[1]
data_path=sys.argv[2]
xid=pickle.load(open(xid_name)) #xid_train.p or xid_test.p

newc=pickle.load(open('newc.p'))

train_or_test=data_path.split('/')[-1]
if train_or_test=='train':
    f=open(data_path[:-5]+'trainLabels.csv')
    f.readline()
    train={}
    for line in f:
        xx=line.split(',')
        train[xx[0][1:-1]]=int(xx[1])  # labels are from 1 -> 9 !
    f.close()
    y=[]

cmd2g={}
for i in newc:
    for j in newc:
        cmd2g[(i,j)]=1
print newc


cmd3g=pickle.load(open('cutcmd3g.p'))
cmd4g=pickle.load(open('cutcmd4g.p'))
head=pickle.load(open('head.p'))

print newc
X2g=[]
X3g=[]
X4g=[]
Xhead=[]
for c,f in enumerate(xid):#(files[len(files)/10*a1:len(files)/10*a2]):
    fo=open(data_path+'/'+f+'.asm')

    count2g={}
    count3g={}
    count4g={}
    for i in cmd2g:
        count2g[i]=0

    for i in cmd3g:
        count3g[i]=0

    for i in cmd4g:
        count4g[i]=0

    counthead={}
    for i in head:
        counthead[i]=0

    tot=0
    a=-1
    b=-1
    d=-1
    e=-1
    for line in fo:

        xx=line.split()

        if xx[0].split(':')[0] in counthead:
            counthead[xx[0].split(':')[0]]+=1

        for x in xx:
            if x in newc:
                a=b
                b=d
                d=e
                e=x
                if (a,b,d,e) in cmd4g:
                    count4g[(a,b,d,e)]+=1
                    tot+=1

                if (b,d,e) in cmd3g:
                    count3g[(b,d,e)]+=1

                if (d,e) in cmd2g:
                    count2g[(d,e)]+=1

    fo.close()
    name=f.split('.')[0]
    if train_or_test=='train': 
        y.append(train[name])
    if True:#c%10000==0:
        print c*1.0/len(xid),tot
    X4g.append([count4g[i] for i in cmd4g])
    X3g.append([count3g[i] for i in cmd3g])
    X2g.append([count2g[i] for i in cmd2g])
    Xhead.append([counthead[i] for i in head])

    del count4g,count2g,count3g,counthead
train_or_test=data_path.split('/')[-1]
pickle.dump(X4g,open('X4g_'+train_or_test+'.p','w'))
pickle.dump(X3g,open('X3g_'+train_or_test+'.p','w'))
pickle.dump(X2g,open('X2g_'+train_or_test+'.p','w'))
pickle.dump(Xhead,open('Xhead_'+train_or_test+'.p','w'))

if train_or_test=='train':
    pickle.dump(y,open('y.p','w'))
