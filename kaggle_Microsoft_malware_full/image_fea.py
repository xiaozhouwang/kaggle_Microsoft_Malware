import numpy,scipy.misc, os, array
def get_feature(data_set = 'train', data_type = 'bytes'):
    files=os.listdir(data_set)
    with open('%s_%s_image.csv'%(data_set, data_type),'wb') as f:
        f.write('Id,%s\n'%','.join(['%s_%i'%(data_type,x)for x in xrange(1000)]))
        for cc,x in enumerate(files):
            if data_type != x.split('.')[-1]:
                continue
            file_id = x.split('.')[0]
            tmp = read_image(data_set + '/' +x)
            f.write('%s,%s\n'%(file_id, ','.join(str(v) for v in tmp)))
            #print "finish..." + file_id
def read_image(filename):
    f = open(filename,'rb')
    ln = os.path.getsize(filename) # length of file in bytes
    width = 256
    rem = ln%width
    a = array.array("B") # uint8 array
    a.fromfile(f,ln-rem)
    f.close()
    g = numpy.reshape(a,(len(a)/width,width))
    g = numpy.uint8(g)
    g.resize((1000,))
    return list(g)

if __name__ == '__main__':
    #get_feature(data_set = 'train', data_type = 'bytes')
    get_feature(data_set = 'train', data_type = 'asm')
    #get_feature(data_set = 'test', data_type = 'bytes')
    get_feature(data_set = 'test', data_type = 'asm')
    print 'DONE asm image features!'

