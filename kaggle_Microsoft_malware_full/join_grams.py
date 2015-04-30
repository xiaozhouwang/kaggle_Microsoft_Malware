import heapq
import pickle
import math
from csv import DictReader
import glob
import os
import csv

def join_ngrams(num = 100000):
    dict_all = dict()
    for c in range(1,10):
        #print "merging %i out of 9"%c
        heap = pickle.load(open('gram/ngram_%i_top%i'%(c,num),'rb'))
        while heap:
            count, gram = heapq.heappop(heap)
            if gram not in dict_all:
                dict_all[gram] = [0]*9
            dict_all[gram][c-1] = count
    return dict_all
    #pickle.dump(dict_all, open('ready_for_selection.pkl','wb'))


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

def Heap_gain(p, n, class_label, dict_all, num_features = 750, gain_minimum_bar = -100000):
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
    return [i[1] for i in heap]

def gen_df(features_all, train = True, verbose = False, N = 4):
    yield ['Id'] + features_all # yield header
    if train == True:
        ds = 'train'
    else:
        ds = 'test'
    directory_names = list(set(glob.glob(os.path.join(ds, "*.bytes"))))
    for f in directory_names:
        f_id = f.split('/')[-1].split('.')[0]
        if verbose == True:
            print 'doing %s'%f_id
        one_list = []
        with open("%s/%s.bytes"%(ds,f_id),'rb') as read_file:
            for line in read_file:
                one_list += line.rstrip().split(" ")[1:]
        grams_string = [''.join(one_list[i:i+N]) for i in xrange(len(one_list)-N)]
        # build a dict for looking up
        
        grams_dict = dict()
        for gram in grams_string:
            if gram not in grams_dict:
                grams_dict[gram] = 1
        
        binary_features = []
        for feature in features_all:
            if feature in grams_dict:
                binary_features.append(1)
            else:
                binary_features.append(0)
        del grams_string
        '''
        ## instead of binary features, do count
        grams_dict = dict()
        for gram in grams_string:
            if gram not in grams_dict:
                grams_dict[gram] = 1
            else:
                grams_dict[gram] += 1 
        
        binary_features = []
        for feature in features_all:
            if feature in grams_dict:
                binary_features.append(grams_dict[feature])
            else:
                binary_features.append(0)
        del grams_string        
        '''
        yield [f_id] + binary_features

if __name__ == '__main__':
    dict_all = join_ngrams()
    features_all = []
    for i in range(1,10):
        p, n = num_instances('trainLabels.csv', i)
        features_all  += Heap_gain(p,n,i,dict_all) # 750 * 9
    train_data = gen_df(features_all, train = True, verbose = False)
    with open('train_data_750.csv','wb') as outfile:
        wr = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in train_data:
            wr.writerow(row)
    test_data = gen_df(features_all, train = False,verbose = False)
    with open('test_data_750.csv','wb') as outfile:
        wr = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in test_data:
            wr.writerow(row)   
    print "DONE 4 gram features!"