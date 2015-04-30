import pickle
import os
import sys

##########################################################
# usage
# pypy find_new_ins.py xid_train.p ./ins_train  ./jump_map_train

# xid_train.p is a list like ['loIP1tiwELF9YNZQjSUO',''....] to specify
# the order of samples in traing data
# ./ins_train is where the local folder of ins.p, {address:ins}
# ./jump_map_train is where the local folder of jump map, {address of this ins: address of next ins}
##########################################################

xid_name=sys.argv[1]
ins_path=sys.argv[2]
jump_map_path=sys.argv[3]

xid=pickle.load(open(xid_name)) #xid_train.p or xid_test.p

cmd={} # new ins found

files=os.listdir(jump_map_path)
mware_that_has_jump={}
for i in files:
    if '.p' in i:
        mware_that_has_jump[i.split('.')[0]]=1

for cc,fx in enumerate(xid):
    tmpcount={}
    ins=pickle.load(open(ins_path+'/'+fx+'.ins.p'))
    insx=[]
    if fx not in mware_that_has_jump: # there is no jump in that malware
        for i in ins:
            if i not in tmpcount:
                tmpcount[i]=0
            tmpcount[i]+=1
        count={}
        for i in tmpcount:
            count[tmpcount[i]]=i
        for j in sorted(count.keys(),reverse=True):
            if j <200:
                break
            if count[j] not in cmd:
                cmd[count[j]]=1 # get the top 200 frequent ins in that mware
        del ins,insx,tmpcount,count
        continue
    jump=pickle.load(open(jump_map_path+'/'+fx+'.p'))
    keys= sorted(ins.keys())
    #print keys[:20]
    nextins={}
    for c,j in enumerate(keys[:-1]):
        if j in jump and jump[j] in ins:
            nextins[j]=jump[j]
            #print j,jump[j]
        else:
            nextins[j]=keys[c+1]
    current=keys[0]
    
    while True:
        if ins[current] not in tmpcount:
            tmpcount[ins[current]]=0
        tmpcount[ins[current]]+=1
        if current not in nextins:
            print 'not in'
            break
        if  sum(tmpcount.values())>len(ins)*5:
            print 'loop runs more than 5x'
            break
        current=nextins[current]

    count={}
    for i in tmpcount:
        count[tmpcount[i]]=i
    for j in sorted(count.keys(),reverse=True):
        if j <200:
            break
        if count[j] not in cmd:
            cmd[count[j]]=0
        cmd[count[j]]+=j
        

    
    del current,ins,insx,jump,keys,nextins,tmpcount,count
              
    print 'find',cc*1.0/len(xid),len(cmd)
print cmd
pickle.dump(cmd,open('newcmd.p','w'))

