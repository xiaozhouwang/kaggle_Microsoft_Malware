import os,array
import pickle
import numpy as np
import sys
xid=pickle.load(open(sys.argv[1]))
asm_code_path=sys.argv[2]
train_or_test=asm_code_path.split('_')[-1]

X = np.zeros((len(xid),2000))
for cc,i in enumerate(xid):
    f=open(asm_code_path+'/'+i+'.asm')
    ln = os.path.getsize(asm_code_path+'/'+i+'.asm') # length of file in bytes
    width = int(ln**0.5)
    rem = ln%width
    a = array.array("B") # uint8 array
    a.fromfile(f,ln-rem)
    f.close()
    a=np.array(a)
    #im = Image.open('asmimage/'+i+'.png')
    a.resize((2000,))
    #im1 = im.resize((64,64),Image.ANTIALIAS); # for faster computation
    #des = leargist.color_gist(im1)
    X[cc] = a#[0,:1000] #des[0:320]
    print cc*1.0/len(xid)
pickle.dump(X,open('Xcode_'+train_or_test+'.p','w'))
