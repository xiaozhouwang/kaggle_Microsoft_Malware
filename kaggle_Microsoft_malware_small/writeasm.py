import pickle
import sys
xid=pickle.load(open(sys.argv[1]))
data_path=sys.argv[2]
asm_code_path=sys.argv[3]

for cc,i in enumerate(xid):
    f=open(data_path+'/'+i+'.asm')
    fo=open(asm_code_path+'/'+i+'.asm','w')
    start=True
    for line in f:
        xx=line.split()
        for c,x in enumerate(xx):
            if x=='Pure':
                if xx[c+1]=='data':
                    start=False
                if xx[c+1]=='code':        
                    start=True
        if True:
            xx[0]=xx[0].split(':')[0]            
            fo.write(''.join(xx)+'\n')
    f.close()
    fo.close()          
    print cc*1.0/len(xid)
