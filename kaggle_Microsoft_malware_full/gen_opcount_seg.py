import subprocess
data_path='.'
opcode_path='op_train'
jump_path='jump_train'
jump_map_path='jump_map_train'

cmd='mkdir '+' '.join([opcode_path,jump_path,jump_map_path])
subprocess.call(cmd,shell=True)

cmd='pypy get_ins.py xid_train.p '+' '.join([data_path+'/train',opcode_path])
subprocess.call(cmd,shell=True)

cmd='pypy get_jump.py xid_train.p '+' '.join([opcode_path,jump_path])
subprocess.call(cmd,shell=True)

cmd='pypy get_jump_map.py xid_train.p '+' '.join([data_path+'/train',jump_path,jump_map_path])
subprocess.call(cmd,shell=True)

cmd='pypy find_new_ins.py xid_train.p '+' '.join([opcode_path,jump_map_path])
subprocess.call(cmd,shell=True)

cmd='pypy filtcmd.py'
subprocess.call(cmd,shell=True)

cmd='pypy find_2g.py xid_train.p '+data_path+'/train'
subprocess.call(cmd,shell=True)

cmd='pypy find_3g.py xid_train.p '+data_path+'/train'
subprocess.call(cmd,shell=True)

cmd='pypy cut3g.py'
subprocess.call(cmd,shell=True)

cmd='pypy cut3g_for_4g.py'
subprocess.call(cmd,shell=True)

cmd='pypy find_4g.py xid_train.p '+data_path+'/train'
subprocess.call(cmd,shell=True)

cmd='pypy cut4g.py'
subprocess.call(cmd,shell=True)

cmd='pypy findhead.py xid_train.p '+data_path+'/train'
subprocess.call(cmd,shell=True)

cmd='pypy rebuild_2g3g4ghead.py xid_train.p '+data_path+'/train'
subprocess.call(cmd,shell=True)

cmd='pypy rebuild_2g3g4ghead.py xid_test.p '+data_path+'/test'
subprocess.call(cmd,shell=True)

cmd='python getfea.py'
subprocess.call(cmd,shell=True)

