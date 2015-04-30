import pickle
import sys
##########################################################
# usage
# pypy getins.py xid_train.p ../../data/train ./ins_train ./jump_train

# xid_train.p is a list like ['loIP1tiwELF9YNZQjSUO',''....] to specify
# the order of samples in traing data
# ../../data/train is the path of original train data
# ./ins_train is where the local folder of ins.p, {address:ins}
# ./jump_train is where the local folder of jmp.p, {address:jump ins}
##########################################################

xid=pickle.load(open(sys.argv[1])) #xid_train.p or xid_test.p
data_path=sys.argv[2]  
ins_path=sys.argv[3]
def isvalid(s):
    Bytes='0123456789ABCDEF'
    if len(s)==2:
        if s[0] in Bytes :
            return False # ins cannot have these
    return True
for cc,fx in enumerate(xid):
    f=open(data_path+'/'+fx+'.asm')
    loc={} # address -> instruction
    for line in f:
        if '.text' != line[:5] and '.code' != line[:5]:
            # most of ins are in those two parts
            continue
        xx=line.split()
        if len(xx)>2:
            add=xx[0].split(':')[1] # address
            for i in xx[1:]:
                if isvalid(i): # get the first token that is not a byte
                    loc[add]=i 
                    break      # one instruction per line (address)
    pickle.dump(loc,open(ins_path+'/'+fx+'.ins.p','w'))
    if cc%50==0:    
        print 'progress',cc*1.0/len(xid),len(loc)
    del loc
    f.close() 
