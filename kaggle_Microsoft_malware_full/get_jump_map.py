import pickle
import sys

##########################################################
# usage
# pypy get_jump_map.py xid_train.p ../../data/train ./jump_train ./jump_map_train

# xid_train.p is a list like ['loIP1tiwELF9YNZQjSUO',''....] to specify
# the order of samples in traing data
# ../../data/train is the path of original train data
# ./instrain is where the local folder of ins.p, {address:ins}
# ./jumptrain is where the local folder of jmp.p, {address:jump ins}
##########################################################



xid=pickle.load(open(sys.argv[1])) #xid_train.p or xid_test.p
data_path=sys.argv[2]
jump_path=sys.argv[3]
jump_map_path=sys.argv[4]
def isvalid_address(s):
    # a legal address should contain only these letters
    letters='0123456789ABCDEF'
    if True:
        for i in s:
            if i not in letters :#or s[1] in words:
                return False
        return True
    return False
cou=0

for cc,fx in enumerate(xid):
    f=open(data_path+'/'+fx+'.asm')
    loc={}  # address jumping dic: start address -> stop address 
    jumpadd=pickle.load(open(jump_path+'/'+fx+'.jmp.p'))
    if len(jumpadd)==0:
        del jumpadd,loc
        continue
    ll=len(jumpadd)
    for line in f:
        if '.text' != line[:5] and '.code' != line[:5]:
            continue
        xx=line.split()
        if len(xx)>2:
            add=xx[0].split(':')[1]  # get address
            if add in jumpadd:  # this is a jump instruction
                for cx,x in enumerate(xx):
                     if x=='jmp' or x=='ja':
                         tid=cx+2  # two patterns: jmp xxx addr or jmp addr
                         if cx+2>=len(xx):
                             tid=cx+1
                         tmpx=xx[tid].split('_') 
                         if len(tmpx)!=2:  # not a valid address
                             break
                         if isvalid_address(tmpx[1]):
                             if len(tmpx[1])<8: # make the address 8 bit
                                 tmpx[1]='0'*(8-len(tmpx[1]))+tmpx[1]
                             loc[add]=tmpx[1]
                             ll=ll-1
                         else:
                             print fx,line#xx[-1].split('_')[1]
                         break
            if ll==0:
                break                
                #print xx[-1][-8:]
    if len(loc)>0:
        pickle.dump(loc,open(jump_map_path+'/'+fx+'.p','w'))
    del loc,jumpadd   
    print cc*1.0/len(xid)
    f.close()
