import sys
import pickle

##########################################################
# usage
# pypy find_4g.py xid_train.p ../../data/train 

# xid_train.p is a list like ['loIP1tiwELF9YNZQjSUO',''....] to specify
# the order of samples in traing data
# ../../data/train is the path of original train data
##########################################################
xid_name=sys.argv[1]
data_path=sys.argv[2]

xid=pickle.load(open(xid_name)) #xid_train.p or xid_test.p

newc=pickle.load(open('newc.p'))
newc2=pickle.load(open('cutcmd3g_for_4g.p'))
cmd4g={}
for i in newc2:
    for j in newc:
        cmd4g[(i[0],i[1],i[2],j)]=0
print newc

for c,f in enumerate(xid):
    count={}
    fo=open(data_path+'/'+f+'.asm')
    tot=0
    a=-1
    b=-1
    d=-1
    e=-1
    for line in fo:
        xx=line.split()
        for x in xx:
            if x in newc:
                
                a=b
                b=d
                d=e
                e=x
                if (a,b,d,e) in cmd4g:
                    if (a,b,d,e) not in count:
                        count[(a,b,d,e)]=0
                    count[(a,b,d,e)]+=1
                    tot+=1
    fo.close()
    if True:#c%10000==0:
        print c*1.0/len(xid),tot
    for i in count:
        cmd4g[i]=count[i]+cmd4g[i]
    del count
cmd4gx={}
for i in cmd4g:
    if cmd4g[i]>0:
        cmd4gx[i]=cmd4g[i]
print len(cmd4gx)
pickle.dump(cmd4gx,open('cmd4g.p','w'))
