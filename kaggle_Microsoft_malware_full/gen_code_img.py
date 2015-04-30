import subprocess
asm_code_path='asm_code_'
data_path='.'
cmd='mkdir '+' '.join([asm_code_path+'train',asm_code_path+'test'])
subprocess.call(cmd,shell=True)

cmd='pypy writeasm.py xid_train.p '+' '.join([data_path+'/train',asm_code_path+'train'])
subprocess.call(cmd,shell=True)

cmd='pypy writeasm.py xid_test.p '+' '.join([data_path+'/test',asm_code_path+'test'])
subprocess.call(cmd,shell=True)

cmd='python rebuild_code.py xid_train.p '+asm_code_path+'train'
subprocess.call(cmd,shell=True)

cmd='python rebuild_code.py xid_test.p '+asm_code_path+'test'
subprocess.call(cmd,shell=True)

