import sys
import pickle

##########################################################
# usage
# pypy find_3g.py xid_train.p ../../data/train 

# xid_train.p is a list like ['loIP1tiwELF9YNZQjSUO',''....] to specify
# the order of samples in traing data
# ../../data/train is the path of original train data
##########################################################
xid_name=sys.argv[1]
data_path=sys.argv[2]

xid=pickle.load(open(xid_name)) #xid_train.p or xid_test.p
newc=pickle.load(open('newc.p'))
newc2=pickle.load(open('cmd2g.p'))

cmd3g={}
for i in newc2:
    for j in newc:
        cmd3g[(i[0],i[1],j)]=0
print newc


for c,f in enumerate(xid):#(files[len(files)/10*a1:len(files)/10*a2]):
    count={}
    #for i in cmd3g:
    #    count[i]=0
    fo=open(data_path+'/'+f+'.asm')
    tot=0
    a=-1
    b=-1
    d=-1
    for line in fo:
        xx=line.split()
        for x in xx:
            if x in newc:
                
                a=b
                b=d
                d=x
                if (a,b,d) in cmd3g:
                    if (a,b,d) not in count:
                        count[(a,b,d)]=0
                    count[(a,b,d)]+=1
                    tot+=1
#                     print (b,a)
    fo.close()
    if True:#c%10000==0:
        print c*1.0/len(xid),tot
    for i in count:
        cmd3g[i]=count[i]+cmd3g[i]
    del count
import pickle
cmd3gx={}
for i in cmd3g:
    if cmd3g[i]>0:
        cmd3gx[i]=cmd3g[i]
print len(cmd3gx)
pickle.dump(cmd3gx,open('cmd3g.p','w'))
