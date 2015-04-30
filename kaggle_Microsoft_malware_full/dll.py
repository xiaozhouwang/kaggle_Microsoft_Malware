import heapq
import pickle
import math
from csv import DictReader
import glob
import os
import csv
from datetime import datetime

# dll call features.

# load file names 
def load_label(path, label):
    result = []
    for row in DictReader(open(path)):
        if int(row['Class']) == label:
            result.append((row['Id']))
    return result

def dll_single_file(f, N = 4):
    pattern_dict = dict()
    f_lines = list()
    with open(f, 'rb') as outfile:
        for line in outfile:
            if 'idata' in line and 'extrn' in line:
                f_lines.append(line.lower())
    for line in f_lines:
        line = line.strip().split()
        p = line[line.index('extrn')+1].split(':')[0]
        if p and p not in pattern_dict:
            pattern_dict[p] = 1
    return pattern_dict

def reduce_dict():
    dict_all = dict()
    for c in range(1,10):
        f_labels = load_label('trainLabels.csv', c)
        for f in f_labels:
            f_name = 'train/'+f+'.asm'
            dll = dll_single_file(f_name)
            for feature in dll:
                if feature not in dict_all:
                    dict_all[feature] = [0]*9
                dict_all[feature][c-1] +=1
        #print "finishing features in class %i"%c
    return dict_all

# load data
def num_instances(path, label):
    p = 0
    n = 0
    for row in DictReader(open(path)):
        if int(row['Class']) == label:
            p += 1
        else:
            n += 1
    return p,n


def entropy(p,n):
    p_ratio = float(p)/(p+n)
    n_ratio = float(n)/(p+n)
    return -p_ratio*math.log(p_ratio) - n_ratio * math.log(n_ratio)

def info_gain(p0,n0,p1,n1,p,n):
    return entropy(p,n) - float(p0+n0)/(p+n)*entropy(p0,n0) - float(p1+n1)/(p+n)*entropy(p1,n1)



def Heap_gain(p, n, class_label, dict_all, num_features = 1000, gain_minimum_bar = -100):
    heap = [(gain_minimum_bar, 'gain_bar')] * num_features
    root = heap[0]
    for gram, count_list in dict_all.iteritems():
        p1 = count_list[class_label-1]
        n1 = sum(count_list[:(class_label-1)] + count_list[class_label:])
        p0,n0 = p - p1, n - n1
        if p1*p0*n1*n0 != 0:
            gain = info_gain(p0,n0,p1,n1,p,n)
            if gain > root[0]:
                root = heapq.heapreplace(heap, (gain, gram))
    #return heap
    result = [i[1] for i in heap if i[1] != 'gain_bar']
    #print "the length of dll for class %i is %i"%(class_label, len(result))
    return result

def gen_df(features_all, train = True, verbose = False, N = 4):
    yield ['Id'] + features_all # yield header
    if train == True:
        ds = 'train'
    else:
        ds = 'test'
    directory_names = list(set(glob.glob(os.path.join(ds, "*.asm"))))
    for f in directory_names:
        f_id = f.split('/')[-1].split('.')[0]
        if verbose == True:
            print 'doing %s'%f_id

        binary_features = list()
        tmp_pattern = dict()
        f_lines = list()
        with open(f, 'rb') as outfile:
            for line in outfile:
                if 'idata' in line and 'extrn' in line:
                    f_lines.append(line.lower())
        for line in f_lines:
            line = line.strip().split()
            p = line[line.index('extrn')+1].split(':')[0]
            if p and p not in tmp_pattern:
                tmp_pattern[p] = 1

        for fea in features_all:
            if fea in tmp_pattern:
                binary_features.append(1)
            else:
                binary_features.append(0)

        yield [f_id] + binary_features

if __name__ == '__main__':
    start = datetime.now()
    dict_all = reduce_dict()
    features_all = []
    for i in range(1,10):
        p, n = num_instances('trainLabels.csv', i)
        features_all  += Heap_gain(p,n,i,dict_all)
    train_data = gen_df(features_all, train = True, verbose = False)
    with open('train_dll.csv','wb') as outfile:
        wr = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in train_data:
            wr.writerow(row)
    test_data = gen_df(features_all, train = False,verbose = False)
    with open('test_dll.csv','wb') as outfile:
        wr = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in test_data:
            wr.writerow(row)
    print "DONE DLL functions!"
    #print datetime.now() - start
