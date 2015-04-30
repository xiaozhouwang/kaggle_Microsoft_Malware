# -*- coding: utf-8 -*-
"""
part of the code borrowed from the benchmark in the forum.
create Frequency Features for 1 byte. So 16*16 features will add to train and test.
"""
from multiprocessing import Pool
import os
from csv import writer


paths = ['train','test']

 
def consolidate(path):

    s_path = path
    Files = os.listdir(s_path)
    byteFiles = [i for i in Files if '.bytes' in i]
    consolidatedFile = path + '_frequency.csv'
    
    with open(consolidatedFile, 'wb') as f:
        # Preparing header part
        fw = writer(f)
        colnames = ['Id']
        colnames += ['FR_'+hex(i)[2:] for i in range(16**2)]
        fw.writerow(colnames)
        
        for t, fname in enumerate(byteFiles):
            consolidation = []
            f = open(s_path+'/'+fname, 'rb')
            twoByte = [0]*16**2
            for row in f:
                codes = row[:-2].split()[1:]
                
                # Conversion of code to to two byte
                twoByteCode = [int(i,16) for i in codes if i != '??']                                     
                # Frequency calculation of two byte codes
                for i in twoByteCode:
                    twoByte[i] += 1
                
            # Row added
            consolidation += [fname[:fname.find('.bytes')]]
            consolidation += twoByte
            
            fw.writerow(consolidation)
            # Writing rows after every 100 files processed
            #if (t+1)%100==0:
            #    print(t+1, 'files loaded for ', path)

if __name__ == '__main__':
    p = Pool(2)
    p.map(consolidate, paths)
    print "DONE bytes count!"
