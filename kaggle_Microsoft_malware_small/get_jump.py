import pickle
import sys

xid=pickle.load(open(sys.argv[1]))
#unconditional_jump=['jmp','j','ja']
ins_path=sys.argv[2]
jump_path=sys.argv[3]

for cc,i in enumerate(xid):
    jmp={}
    tmp=pickle.load(open(ins_path+'/'+i+'.ins.p'))
    for add in tmp:
        if tmp[add] == 'jmp' or tmp[add]=='ja':
            jmp[add]=1
    del tmp
    pickle.dump(jmp,open(jump_path+'/'+i+'.jmp.p','w'))
    del jmp

    print cc*1.0/len(xid)
















