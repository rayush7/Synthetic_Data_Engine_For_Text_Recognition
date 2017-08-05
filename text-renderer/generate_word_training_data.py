#a Max Jaderberg 16/5/14
# Generates training data using WordRenderer
import sys
import os
from titan_utils import is_cluster, get_task_id, crange
from word_renderer import WordRenderer, FontState, FileCorpus, TrainingCharsColourState, SVTFillImageState, wait_key, NgramCorpus, RandomCorpus
from scipy.io import savemat
from PIL import Image
import numpy as n
import tarfile
import h5py
import pandas as pd
import lmdb
import cv2


SETTINGS = {
    #####################################
    'RAND10': {
        'corpus_class': RandomCorpus,
        'corpus_args': {'min_length': 1, 'max_length': 10},
        'fontstate':{
            'font_list': ["/home/ubuntu/Datasets/SVT/font_path_list.txt",
                      "/home/ubuntu/Datasets/SVT/font_path_list.txt"],
            'random_caps': 1,  # the corpus is NOT case sensitive so train with all sorts of caps
        },
        'trainingchars_fn': ["/home/ubuntu/Datasets/SVT/icdar_2003_train.txt",
                             "/home/ubuntu/Datasets/SVT/icdar_2003_train.txt"],
        'fillimstate': {
            'data_dir': ["/home/ubuntu/Datasets/SVT/svt/svt1/img",
                         "/home/ubuntu/Datasets/SVT/svt/svt1/img"],
            'gtmat_fn': ["/home/ubuntu/Datasets/SVT/svt.txt",
                         "/home/ubuntu/Datasets/SVT/svt.txt"],
        }
    },
    #####################################
    'RAND23': {
        'corpus_class': RandomCorpus,
        'corpus_args': {'min_length': 1, 'max_length': 23},
        'fontstate':{
            'font_list': ["/home/ubuntu/Datasets/SVT/font_path_list.txt",
                      "/home/ubuntu/Datasets/SVT/font_path_list.txt"],
            'random_caps': 1,  # the corpus is NOT case sensitive so train with all sorts of caps
        },
        'trainingchars_fn': ["/home/ubuntu/Datasets/SVT/icdar_2003_train.txt",
                             "/home/ubuntu/Datasets/SVT/icdar_2003_train.txt"],
        'fillimstate': {
            'data_dir': ["/home/ubuntu/Datasets/SVT/svt/svt1/img",
                         "/home/ubuntu/Datasets/SVT/svt/svt1/img"],
            'gtmat_fn': ["/home/ubuntu/Datasets/SVT/svt.txt",
                         "/home/ubuntu/Datasets/SVT/svt.txt"],
        }
    },
}


#------------GET Labels--------------------------------------------------

def get_labels(input_list):
    out_list=[]
    
    for x in input_list:
        names=os.path.basename(x)
        res=names.partition('_')[2].partition('_')[0]
        out_list.append(res)
        
    return out_list


#---------------CREATING LMDB DATASET--------------------------------------------
def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = n.fromstring(imageBin, dtype=n.uint8)

    if imageBuf.size==0:
        return False

    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)
            #txn.put(k.encode(),v.encode())


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

#---------------------CREATING TRAINING DATA-----------

def create_synthetic_data(lmdb_path,imfolder_path,dataset,NUM_TO_GENERATE):

    #NUM_PER_FOLDER = 10 #1000
    SAMPLE_HEIGHT = 32
    QUALITY = [80, 10]

    iscluster = int(is_cluster())

    settings = SETTINGS[dataset]

    if not os.path.exists(imfolder_path):
    	os.makedirs(imfolder_path)

    ngram_mode = settings.get('ngram_mode', False)

    # init providers
    if 'corpus_class' in settings:
        corp_class = settings['corpus_class']
    else:
        corp_class = FileCorpus
    if 'corpus_args' in settings:
        corpus = corp_class(settings['corpus_args'])
    else:
        corpus = corp_class()
    fontstate = FontState(font_list=settings['fontstate']['font_list'][iscluster])
    fontstate.random_caps = settings['fontstate']['random_caps']
    colourstate = TrainingCharsColourState(settings['trainingchars_fn'][iscluster])
    if not isinstance(settings['fillimstate'], list):
        fillimstate = SVTFillImageState(settings['fillimstate']['data_dir'][iscluster], settings['fillimstate']['gtmat_fn'][iscluster])
    else:
        # its a list of different fillimstates to combine
        states = []
        for i, fs in enumerate(settings['fillimstate']):
            s = SVTFillImageState(fs['data_dir'][iscluster], fs['gtmat_fn'][iscluster])
            # move datadir to imlist
            s.IMLIST = [os.path.join(s.DATA_DIR, l) for l in s.IMLIST]
            states.append(s)
        fillimstate = states.pop()
        for fs in states:
            fillimstate.IMLIST.extend(fs.IMLIST)

    # take substrings
    try:
        substr_crop = settings['substrings']
    except KeyError:
        substr_crop = -1

    # init renderer
    sz = (800,200)
    WR = WordRenderer(sz=sz, corpus=corpus, fontstate=fontstate, colourstate=colourstate, fillimstate=fillimstate)

    count=0

    #Declating the Image_Name List and Label List
    im_list=[]
    label_list=[]


    for i in crange(range(0, NUM_TO_GENERATE)):

        # gen sample
        try:
            data = WR.generate_sample(outheight=SAMPLE_HEIGHT, random_crop=True, substring_crop=substr_crop, char_annotations=(substr_crop>0))
        except Exception:
            print "\tERROR"
            continue
        if data is None:
            print "\tcould not generate good sample"
            continue

        if not ngram_mode:
            fnstart = "%s_%s_%d" % ('synthetic', data['text'], data['label'])
        else:
            fnstart = "%s_%s_%d" % ('synthetic', data['text'], data['label']['word_label'])

        # save with random compression
        quality = min(80, max(0, int(QUALITY[1]*n.random.randn() + QUALITY[0])))
        try:
            img = Image.fromarray(data['image'])
        except Exception:
            print "\tbad image generated"
            continue

        if img.mode != 'RGB':
            img = img.convert('RGB')
 #       imfn = os.path.join(imfolder_path, fnstart + ".jpg")
        
        img.save(imfolder_path+str(count)+'.jpg','JPEG',quality=quality)
        print 'Creating Image :', count

        # Save Data for LMDB
        im_list.append(str(count)+'.jpg')
        label_list.append(data['text'])

        count=count+1

    #Saving the Dataframe
    im_list = [imfolder_path+x for x in im_list]

    print 'Length of Image Path List: ', len(im_list)
    print 'Length of Image Label List: ', len(label_list)

    print im_list[0]
    print type(im_list[0])
    print label_list
    print type(label_list[0])

    #df_synthetic=pd.DataFrame(columns=['Image_Path','Image_Label'])
    #df_synthetic['Image_Path']=im_list
    #df_synthetic['Image_Label']=label_list
    #df_synthetic.to_csv('Synthetic_data_info.csv',sep='\t',index=None)

    #Creating LMDB Dataset using create_dataset function

    print 'Creating LMDB Dataset'
    createDataset(lmdb_path, im_list, label_list, lexiconList=None, checkValid=True)

    print 'Finished creating LMDB Datasets'

def main():

	#Image Folder Path
	train_im_folder_path='/home/ubuntu/Datasets/text-renderer/vgg_synthetic_custom_train/'
	val_im_folder_path='/home/ubuntu/Datasets/text-renderer/vgg_synthetic_custom_val/'

	#Setting LMDB Folder Path
	train_lmdb_path='/home/ubuntu/Datasets/text-renderer/synth90k_custom_train_lmdb'
	val_lmdb_path='/home/ubuntu/Datasets/text-renderer/synth90k_custom_val_lmdb'

	#Number of Training and Val Images to Generate
	NUM_TO_GENERATE_TRAIN = 250000
	NUM_TO_GENERATE_VAL = 50000

	#Type of Data to Generate
	dataset_type='RAND10'

	#Creating the Training Data
	print 'NUM_TO_GENERATE_TRAIN',NUM_TO_GENERATE_TRAIN
	create_synthetic_data(train_lmdb_path,train_im_folder_path,dataset_type,NUM_TO_GENERATE_TRAIN)

	#Creating the Validation Data
	print 'NUM_TO_GENERATE_VAL',NUM_TO_GENERATE_VAL
	create_synthetic_data(val_lmdb_path,val_im_folder_path,dataset_type,NUM_TO_GENERATE_VAL)

	print "FINISHED! Creating Training and Validation Data"


if __name__ == '__main__':
	main()
