# -*- coding: utf-8 -*-
## instructions frequency

from multiprocessing import Pool
import os
import csv

paths = ['train','test']

instr_set = set(['mov','xchg','stc','clc','cmc','std','cld','sti','cli','push',
	'pushf','pusha','pop','popf','popa','cbw','cwd','cwde','in','out',
	'add','adc','sub','sbb','div','idiv','mul','imul','inc','dec',
	'cmp','sal','sar','rcl','rcr','rol','ror','neg','not','and'
	'or','xor','shl','shr','nop','lea','int','call','jmp',
	'je','jz','jcxz','jp','jpe','ja','jae','jb','jbe','jna',
	'jnae','jnb','jnbe','jc','jnc','ret','jne','jnz','jecxz',
	'jnp','jpo','jg','jge','jl','jle','jng','jnge','jnl','jnle',
	'jo','jno','js','jns'])

def consolidate(path,instr_set = instr_set):
	Files = os.listdir(path)
	asmFiles = [i for i in Files if '.asm' in i]
	consolidatedFile = path + '_instr_frequency.csv'
	with open(consolidatedFile, 'wb') as f:
		fieldnames = ['Id'] + list(instr_set)
		writer = csv.DictWriter(f, fieldnames = fieldnames)
		writer.writeheader()
		for t, fname in enumerate(asmFiles):
			consolidation = dict(zip(instr_set,[0]*len(instr_set)))
			consolidation['Id'] = fname[:fname.find('.asm')]
			with open(path+'/'+fname, 'rb') as f:
				for line in f:
					if 'text' in line and ',' in line and ';' not in line:
						row = line.lower().strip().split('  ')[1:]
						if row:
							tmp_list = [x.strip() for x in row if x != '']
							if len(tmp_list) == 2 and tmp_list[0] in consolidation:
								consolidation[tmp_list[0]] += 1
			writer.writerow(consolidation)
			#if (t+1)%100 == 0:
			#	print str(t+1) + 'files loaded for ' + path

if __name__ == '__main__':
	p = Pool(2)
	p.map(consolidate, paths)
	print "DONE instruction count!"



