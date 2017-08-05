# Max Jaderberg 12/5/14
# Module for rendering words
# BLEND MODES: http://stackoverflow.com/questions/601776/what-do-the-blend-modes-in-pygame-mean
# Rendered words have three colours - base char (0) and border/shadow (128) and background (255)

# TO RUN ON TITAN: use source ~/Work/virtual_envs/paintings_env/bin/activate

import sys
import pygame
import os
import re
from pygame.locals import *
import numpy as n
from pygame import freetype
import math
from matplotlib import pyplot
from PIL import Image
from scipy import ndimage, interpolate
import scipy.cluster
from matplotlib import cm
import random
from scipy.io import loadmat
import time
import h5py
import pandas as pd
import cv2

def wait_key():
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_SPACE:
                return

def rgb2gray(rgb):
    # RGB -> grey-scale (as in Matlab's rgb2grey)
    try:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    except IndexError:
        try:
            gray = rgb[:,:,0]
        except IndexError:
            gray = rgb[:,:]
    return gray

def resize_image(im, r=None, newh=None, neww=None, filtering=Image.BILINEAR):
    dt = im.dtype
    I = Image.fromarray(im)
    if r is not None:
        h = im.shape[0]
        w = im.shape[1]
        newh = int(round(r*h))
        neww = int(round(r*w))
    if neww is None:
        neww = int(newh*im.shape[1]/float(im.shape[0]))
    if newh > im.shape[0]:
        I = I.resize([neww, newh], Image.ANTIALIAS)
    else:
        I.thumbnail([neww, newh], filtering)
    return n.array(I).astype(dt)

def matrix_mult(A, B):
    C = n.empty((A.shape[0], B.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i,j] = n.sum(A[i,:]*B[:,j])
    return C

def save_screen_img(pg_surface, fn, quality=100):
    imgstr = pygame.image.tostring(pg_surface, 'RGB')
    im = Image.fromstring('RGB', pg_surface.get_size(), imgstr)
    im.save(fn, quality=quality)
    print fn

MJBLEND_NORMAL = "normal"
MJBLEND_ADD = "add"
MJBLEND_SUB = "subtract"
MJBLEND_MULT = "multiply"
MJBLEND_MULTINV = "multiplyinv"
MJBLEND_SCREEN = "screen"
MJBLEND_DIVIDE = "divide"
MJBLEND_MIN = "min"
MJBLEND_MAX = "max"

def grey_blit(src, dst, blend_mode=MJBLEND_NORMAL):
    """
    This is for grey + alpha images
    """
    # http://stackoverflow.com/a/3375291/190597
    # http://stackoverflow.com/a/9166671/190597
    # blending with alpha http://stackoverflow.com/questions/1613600/direct3d-rendering-2d-images-with-multiply-blending-mode-and-alpha
    # blending modes from: http://www.linuxtopia.org/online_books/graphics_tools/gimp_advanced_guide/gimp_guide_node55.html
    dt = dst.dtype
    src = src.astype(n.single)
    dst = dst.astype(n.single)
    out = n.empty(src.shape, dtype = 'float')
    alpha = n.index_exp[:, :, 1]
    rgb = n.index_exp[:, :, 0]
    src_a = src[alpha]/255.0
    dst_a = dst[alpha]/255.0
    out[alpha] = src_a+dst_a*(1-src_a)
    old_setting = n.seterr(invalid = 'ignore')
    src_pre = src[rgb]*src_a
    dst_pre = dst[rgb]*dst_a
    # blend:
    blendfuncs = {
        MJBLEND_NORMAL: lambda s, d, sa_: s + d*sa_,
        MJBLEND_ADD: lambda s, d, sa_: n.minimum(255, s + d),
        MJBLEND_SUB: lambda s, d, sa_: n.maximum(0, s - d),
        MJBLEND_MULT: lambda s, d, sa_: s*d*sa_ / 255.0,
        MJBLEND_MULTINV: lambda s, d, sa_: (255.0 - s)*d*sa_ / 255.0,
        MJBLEND_SCREEN: lambda s, d, sa_: 255 - (1.0/255.0)*(255.0 - s)*(255.0 - d*sa_),
        MJBLEND_DIVIDE: lambda s, d, sa_: n.minimum(255, d*sa_*256.0 / (s + 1.0)),
        MJBLEND_MIN: lambda s, d, sa_: n.minimum(d*sa_, s),
        MJBLEND_MAX: lambda s, d, sa_: n.maximum(d*sa_, s),
    }
    out[rgb] = blendfuncs[blend_mode](src_pre, dst_pre, (1-src_a))
    out[rgb] /= out[alpha]
    n.seterr(**old_setting)
    out[alpha] *= 255
    n.clip(out,0,255)
    # astype('uint8') maps np.nan (and np.inf) to 0
    out = out.astype(dt)
    return out

class Corpus(object):
    """
    Defines a corpus of words
    """
   # valid_ascii = [48,49,50,51,52,53,54,55,56,57,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90]
    valid_ascii = [36,37,42,43,45,46,47,48,49,50,51,52,53,54,55,56,57,58,61,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122]

    def __init__(self):
        pass

class TestCorpus(Corpus):
    """
    Just a test corpus from a text file
    """
    CORPUS_FN = "./corpus.txt"

    def __init__(self, args={'unk_probability': 0}):
        self.corpus_text = ""
        pattern = re.compile('[^a-zA-Z0-9 ]')
        for line in open(self.CORPUS_FN):
            line = line.replace('\n', ' ')
            line = pattern.sub('', line)
            self.corpus_text = self.corpus_text + line
        self.corpus_text = ''.join(c for c in self.corpus_text if c.isalnum() or c.isspace())
        self.corpus_list = self.corpus_text.split()
        self.unk_probability = args['unk_probability']

    def get_sample(self, length=None):
        """
        Return a word sample from the corpus, with optional length requirement (cropped word)
        """
        sampled = False
        idx = n.random.randint(0, len(self.corpus_list))
        breakamount = 1000
        count = 0
        while not sampled:
            samp = self.corpus_list[idx]
            if length > 0:
                if len(samp) >= length:
                    if len(samp) > length:
                        # start at a random point in this word
                        diff = len(samp) - length
                        starti = n.random.randint(0, diff)
                        samp = samp[starti:starti+length]
                    break
            else:
                break
            idx = n.random.randint(0, len(self.corpus_list))
            count += 1
            if count > breakamount:
                raise Exception("Unable to generate a good corpus sample")
        if n.random.rand() < self.unk_probability:
            # change some letters to make it random
            ntries = 0
            while True:
                ntries += 1
                if len(samp) > 2:
                    n_to_change = n.random.randint(2, len(samp))
                else:
                    n_to_change = max(1, len(samp) - 1)
                idx_to_change = n.random.permutation(len(samp))[0:n_to_change]
                samp = list(samp)
                for idx in idx_to_change:
                    samp[idx] = chr(random.choice(self.valid_ascii))
                samp = "".join(samp)
                if samp not in self.corpus_list:
                    idx = len(self.corpus_list)
                    break
                if ntries > 10:
                    idx = self.corpus_list.index(samp)
                    break
        return samp, idx

class SVTCorpus(TestCorpus):
    CORPUS_FN = "/Users/jaderberg/Data/TextSpotting/DataDump/svt1/svt_lex_lower.txt"

class FileCorpus(TestCorpus):
    def __init__(self, args):
        self.CORPUS_FN = args['fn']
        TestCorpus.__init__(self, args)

class NgramCorpus(TestCorpus):
    """
    Spits out a word sample, dictionary label, and ngram encoding labels
    """
    def __init__(self, args):
        words_fn = args['encoding_fn_base'] + '_words.txt'
        idx_fn = args['encoding_fn_base'] + '_idx.txt'
        values_fn = args['encoding_fn_base'] + '_values.txt'

        self.words = self._load_list(words_fn)
        self.idx = self._load_list(idx_fn, split=' ', tp=int)
        self.values = self._load_list(values_fn, split=' ', tp=int)

    def get_sample(self, length=None):
        """
        Return a word sample from the corpus, with optional length requirement (cropped word)
        """
        sampled = False
        idx = n.random.randint(0, len(self.words))
        breakamount = 1000
        count = 0
        while not sampled:
            samp = self.words[idx]
            if length > 0:
                if len(samp) >= length:
                    if len(samp) > length:
                        # start at a random point in this word
                        diff = len(samp) - length
                        starti = n.random.randint(0, diff)
                        samp = samp[starti:starti+length]
                    break
            else:
                break
            idx = n.random.randint(0, len(self.words))
            count += 1
            if count > breakamount:
                raise Exception("Unable to generate a good corpus sample")

        return samp, {
            'word_label': idx,
            'ngram_labels': self.idx[idx],
            'ngram_counts': self.values[idx],
        }

    def _load_list(self, listfn, split=None, tp=str):
        arr = []
        for l in open(listfn):
            l = l.strip()
            if split is not None:
                l = [tp(x) for x in l.split(split)]
            else:
                l = tp(l)
            arr.append(l)
        return arr

class RandomCorpus(Corpus):
    """
    Generates random strings
    """
    def __init__(self, args={'min_length': 1, 'max_length': 23}):
        self.min_length = args['min_length']
        self.max_length = args['max_length']

    def get_sample(self, length=None):
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        samp = ""
        for i in range(length):
            samp = samp + chr(random.choice(self.valid_ascii))
        return samp, length



class FontState(object):
    """
    Defines the random state of the font rendering
    """
    size = [60, 10]  # normal dist mean, std
    underline = 0.05
    strong = 0.5
    oblique = 0.2
    wide = 0.5
    strength = [0.02778, 0.05333]  # uniform dist in this interval
    underline_adjustment = [1.0, 2.0]  # normal dist mean, std
    kerning = [2, 5, 0, 20]  # beta distribution alpha, beta, offset, range (mean is a/(a+b))
    border = 0.25
    random_caps = 1.0
    capsmode = [str.lower, str.upper, str.capitalize]  # lower case, upper case, proper noun
    curved = 0.2
    random_kerning = 0.2
    random_kerning_amount = 0.1

    def __init__(self, font_list="/home/ubuntu/Datasets/SVT/font_path_list.txt"):
        self.FONT_LIST = font_list
        base_dir = '/'.join(self.FONT_LIST.split('/')[:-1])
        self.fonts = [os.path.join(base_dir, f.strip()) for f in open(self.FONT_LIST)]

    def get_sample(self):
        """
        Samples from the font state distribution
        """
        return {
            'font': self.fonts[int(n.random.randint(0, len(self.fonts)))],
            'size': self.size[1]*n.random.randn() + self.size[0],
            'underline': n.random.rand() < self.underline,
            'underline_adjustment': max(2.0, min(-2.0, self.underline_adjustment[1]*n.random.randn() + self.underline_adjustment[0])),
            'strong': n.random.rand() < self.strong,
            'oblique': n.random.rand() < self.oblique,
            'strength': (self.strength[1] - self.strength[0])*n.random.rand() + self.strength[0],
            'char_spacing': int(self.kerning[3]*(n.random.beta(self.kerning[0], self.kerning[1])) + self.kerning[2]),
            'border': n.random.rand() < self.border,
            'random_caps': n.random.rand() < self.random_caps,
            'capsmode': random.choice(self.capsmode),
            'curved': n.random.rand() < self.curved,
            'random_kerning': n.random.rand() < self.random_kerning,
            'random_kerning_amount': self.random_kerning_amount,
        }

class AffineTransformState(object):
    """
    Defines the random state for an affine transformation
    """
    proj_type = Image.AFFINE
    rotation = [0, 5]  # rotate normal dist mean, std
    skew = [0, 0]  # skew normal dist mean, std

    def sample_transformation(self, imsz):
        theta = math.radians(self.rotation[1]*n.random.randn() + self.rotation[0])
        ca = math.cos(theta)
        sa = math.sin(theta)
        R = n.zeros((3,3))
        R[0,0] = ca
        R[0,1] = -sa
        R[1,0] = sa
        R[1,1] = ca
        R[2,2] = 1
        S = n.eye(3,3)
        S[0,1] = math.tan(math.radians(self.skew[1]*n.random.randn() + self.skew[0]))
        A = matrix_mult(R,S)
        x = imsz[1]/2
        y = imsz[0]/2
        return (A[0,0], A[0,1], -x*A[0,0] - y*A[0,1] + x,
            A[1,0], A[1,1], -x*A[1,0] - y*A[1,1] + y)

class PerspectiveTransformState(object):
    """
    Defines teh random state for a perspective transformation
    Might need to use http://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    """
    proj_type = Image.PERSPECTIVE
    a_dist = [1, 0.01]
    b_dist = [0, 0.005]
    c_dist = [0, 0.005]
    d_dist = [1, 0.01]
    e_dist = [0, 0.0005]
    f_dist = [0, 0.0005]

    def v(self, dist):
        return dist[1]*n.random.randn() + dist[0]

    def sample_transformation(self, imsz):
        x = imsz[1]/2
        y = imsz[0]/2
        a = self.v(self.a_dist)
        b = self.v(self.b_dist)
        c = self.v(self.c_dist)
        d = self.v(self.d_dist)
        e = self.v(self.e_dist)
        f = self.v(self.f_dist)

        # scale a and d so scale kept same
        #a = 1 - e*x
        #d = 1 - f*y

        z = -e*x - f*y + 1
        A = n.zeros((3,3))
        A[0,0] = a + e*x
        A[0,1] = b+f*x
        A[0,2] = -a*x-b*y-e*x*x-f*x*y+x
        A[1,0] = c+e*y
        A[1,1] = d+f*y
        A[1,2] = -c*x-d*y-e*x*y-f*y*y+y
        A[2,0] = e
        A[2,1] = f
        A[2,2] = z
        # print a,b,c,d,e,f
        # print z
        A = A / z

        return (A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2], A[2,0], A[2,1])

class ElasticDistortionState(object):
    """
    Defines a random state for elastic distortions
    """
    displacement_range = 1
    alpha_dist = [[15, 30], [0, 2]]
    sigma = [[8, 2], [0.2, 0.2]]
    min_sigma = [4, 0]

    def sample_transformation(self, imsz):
        choices = len(self.alpha_dist)
        c = int(n.random.randint(0, choices))
        sigma = max(self.min_sigma[c], n.abs(self.sigma[c][1]*n.random.randn() + self.sigma[c][0]))
        alpha = n.random.uniform(self.alpha_dist[c][0], self.alpha_dist[c][1])
        dispmapx = n.random.uniform(-1*self.displacement_range, self.displacement_range, size=imsz)
        dispmapy = n.random.uniform(-1*self.displacement_range, self.displacement_range, size=imsz)
        dispmapx = alpha * ndimage.gaussian_filter(dispmapx, sigma)
        dispmaxy = alpha * ndimage.gaussian_filter(dispmapy, sigma)
        return dispmapx, dispmaxy

class BorderState(object):
    outset = 0.5
    width = [4, 4]  # normal dist
    position = [[0,0], [-1,-1], [-1,1], [1,1], [1,-1]]

    def get_sample(self):
        p = self.position[int(n.random.randint(0,len(self.position)))]
        w = max(1, int(self.width[1]*n.random.randn() + self.width[0]))
        return {
            'outset': n.random.rand() < self.outset,
            'width': w,
            'position': [int(-1*n.random.uniform(0,w*p[0]/1.5)), int(-1*n.random.uniform(0,w*p[1]/1.5))]
        }

class ColourState(object):
    """
    Gives the foreground, background, and optionally border colourstate.
    Does this by sampling from a training set of images, and clustering in to desired number of colours
    (http://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image)
    """
    IMFN = "/home/ubuntu/Datasets/text-renderer/image_24_results.png"

    def __init__(self):
        self.im = rgb2gray(n.array(Image.open(self.IMFN)))

    def get_sample(self, n_colours):
        #print 'Inside Color State'
        a = self.im.flatten()
        codes, dist = scipy.cluster.vq.kmeans(a, n_colours)
        # get std of centres
        vecs, dist = scipy.cluster.vq.vq(a, codes)

        colours = []
        for i in range(n_colours):
            try:
                code = codes[i]
                std = n.std(a[vecs==i])
                colours.append(std*n.random.randn() + code)
            except IndexError:
                print "\tcolour error"
                colours.append(int(sum(colours)/float(len(colours))))
        # choose randomly one of each colour
        return n.random.permutation(colours)

class TrainingCharsColourState(object):
    """
    Gives the foreground, background, and optionally border colourstate.
    Does this by sampling from a training set of images, and clustering in to desired number of colours
    (http://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image)
    """

    def __init__(self, matfn="/home/ubuntu/Datasets/SVT/icdar_2003_train.txt"):
        #self.ims = loadmat(matfn)['images']
         #with open(matfn) as f: self.ims = f.read().splitlines()
         list_fn=list(pd.read_csv(matfn,sep='\t')['Image_Path'])
         self.ims=list_fn

    def get_sample(self, n_colours):
        curs = 0
        while True:
            curs += 1
            if curs > 1000:
                print "problem with colours"
                break

            #im_sample=cv2.imread(random.choice(self.ims),0)
            #im = cv2.normalize(im_sample.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) # Convert to normalized floating point
            imfn = random.choice(self.ims)
            im = rgb2gray(n.array(Image.open(imfn)))


            #im = self.ims[...,n.random.randint(0, self.ims.shape[2])]

            a = im.flatten()
            codes, dist = scipy.cluster.vq.kmeans(a, n_colours)
            if len(codes) != n_colours:
                continue
            # get std of centres
            vecs, dist = scipy.cluster.vq.vq(a, codes)
            colours = []
            for i, code in enumerate(codes):
                std = n.std(a[vecs==i])
                colours.append(std*n.random.randn() + code)
            break
        # choose randomly one of each colour
        return n.random.permutation(colours)

class FillImageState(object):
    """
    Handles the images used for filling the background, foreground, and border surfaces
    """
    DATA_DIR = '/home/ubuntu/Pictures/'
    IMLIST = ['maxresdefault.jpg', 'alexis-sanchez-arsenal-wallpaper-phone.jpg', 'alexis.jpeg']
    blend_amount = [0.0, 0.25]  # normal dist mean, std
    blend_modes = [MJBLEND_NORMAL, MJBLEND_ADD, MJBLEND_MULTINV, MJBLEND_SCREEN, MJBLEND_MAX]
    blend_order = 0.5
    min_textheight = 16.0  # minimum pixel height that you would find text in an image

    def get_sample(self, surfarr):
        """
        The image sample returned should not have it's aspect ratio changed, as this would never happen in real world.
        It can still be resized of course.
        """
        # load image
        imfn = os.path.join(self.DATA_DIR, random.choice(self.IMLIST))
        baseim = n.array(Image.open(imfn))



        # choose a colour channel or rgb2gray
        if baseim.ndim == 3:
            if n.random.rand() < 0.25:
                baseim = rgb2gray(baseim)
            else:
                baseim = baseim[...,n.random.randint(0,3)]
        else:
            assert(baseim.ndim == 2)

        imsz = baseim.shape
        surfsz = surfarr.shape





        # don't resize bigger than if at the original size, the text was less than min_textheight
        max_factor = float(surfsz[0])/self.min_textheight
        # don't resize smaller than it is smaller than a dimension of the surface
        min_factor = max(float(surfsz[0] + 5)/float(imsz[0]), float(surfsz[1] + 5)/float(imsz[1]))
        # sample a resize factor
        factor = max(min_factor, min(max_factor, ((max_factor-min_factor)/1.5)*n.random.randn() + max_factor))
        sampleim = resize_image(baseim, factor)
        imsz = sampleim.shape
        # sample an image patch
        good = False
        curs = 0
        while not good:
            curs += 1
            if curs > 1000:
                print "difficulty getting sample"
                break
            try:
                x = n.random.randint(0,imsz[1]-surfsz[1])
                y = n.random.randint(0,imsz[0]-surfsz[0])
                good = True
            except ValueError:
                # resample factor
                factor = max(min_factor, min(max_factor, ((max_factor-min_factor)/1.5)*n.random.randn() + max_factor))
                sampleim = resize_image(baseim, factor)
                imsz = sampleim.shape
        imsample = (n.zeros(surfsz) + 255).astype(surfarr.dtype)
        imsample[...,0] = sampleim[y:y+surfsz[0],x:x+surfsz[1]]
        imsample[...,1] = surfarr[...,1].copy()

        return {
            'image': imsample,
            'blend_mode': random.choice(self.blend_modes),
            'blend_amount': min(1.0, n.abs(self.blend_amount[1]*n.random.randn() + self.blend_amount[0])),
            'blend_order': n.random.rand() < self.blend_order,
        }

#class SVTFillImageState(FillImageState):
#    def __init__(self, data_dir, gtmat_fn):
#        self.DATA_DIR = data_dir
#        gtmat = loadmat(gtmat_fn)['gt']
#        with open(gtmat_fn) as f: self.IMLIST = f.read().splitlines()

class SVTFillImageState(FillImageState):
    IMLIST=[]
    def __init__(self,data_dir,gtmat_fn):
        self.DATA_DIR=data_dir
        list_fn=list(pd.read_csv(gtmat_fn,sep='\t')['Image_Path'])
        self.IMLIST=list_fn
        #print list_fn
        #for gtmat_fn in list_fn:
        #    with open(gtmat_fn) as f:
        #        self.IMLIST += (f.read().splitlines())
        #        print self.IMLIST

class DistortionState(object):
    blur = [0, 1]
    sharpen = 0
    sharpen_amount = [30, 10]
    noise = 4
    resample = 0.1
    resample_range = [24, 32]

    def get_sample(self):
        return {
            'blur': n.abs(self.blur[1]*n.random.randn() + self.blur[0]),
            'sharpen': n.random.rand() < self.sharpen,
            'sharpen_amount': self.sharpen_amount[1]*n.random.randn() + self.sharpen_amount[0],
            'noise': self.noise,
            'resample': n.random.rand() < self.resample,
            'resample_height': int(n.random.uniform(self.resample_range[0], self.resample_range[1]))
        }

class SurfaceDistortionState(DistortionState):
    noise = 8
    resample = 0


class BaselineState(object):
    curve = lambda this, a: lambda x: a*x*x
    differential = lambda this, a: lambda x: 2*a*x
    a = [0, 0.1]

    def get_sample(self):
        """
        Returns the functions for the curve and differential for a and b
        """
        a = self.a[1]*n.random.randn() + self.a[0]
        return {
            'curve': self.curve(a),
            'diff': self.differential(a),
        }


class WordRenderer(object):

    def __init__(self, sz=(800,200), corpus=TestCorpus, fontstate=FontState, colourstate=ColourState, fillimstate=FillImageState):
        # load corpus
        self.corpus = corpus() if isinstance(corpus,type) else corpus
        # load fonts
        self.fontstate = fontstate() if isinstance(fontstate,type) else fontstate
        # init renderer
        pygame.init()
        self.sz = sz
        self.screen = None

        self.perspectivestate = PerspectiveTransformState()
        self.affinestate = AffineTransformState()
        self.borderstate = BorderState()
        self.colourstate = colourstate() if isinstance(colourstate,type) else colourstate
        self.fillimstate = fillimstate() if isinstance(fillimstate,type) else fillimstate
        self.diststate = DistortionState()
        self.surfdiststate = SurfaceDistortionState()
        self.baselinestate = BaselineState()
        self.elasticstate = ElasticDistortionState()


    def invert_surface(self, surf):
        pixels = pygame.surfarray.pixels2d(surf)
        pixels ^= 2 ** 32 - 1
        del pixels

    def invert_arr(self, arr):
        arr ^= 2 ** 32 - 1
        return arr

    def apply_perspective_surf(self, surf):
        self.invert_surface(surf)
        data = pygame.image.tostring(surf, 'RGBA')
        img = Image.fromstring('RGBA', surf.get_size(), data)
        img = img.transform(img.size, self.affinestate.proj_type,
            self.affinestate.sample_transformation(img.size),
            Image.BICUBIC)
        img = img.transform(img.size, self.perspectivestate.proj_type,
            self.perspectivestate.sample_transformation(img.size),
            Image.BICUBIC)
        im = n.array(img)
        # pyplot.imshow(im)
        # pyplot.show()
        surf = pygame.surfarray.make_surface(im[...,0:3].swapaxes(0,1))
        self.invert_surface(surf)
        return surf

    def apply_perspective_arr(self, arr, affstate, perstate, filtering=Image.BICUBIC):
        img = Image.fromarray(arr)
        img = img.transform(img.size, self.affinestate.proj_type,
            affstate,
            filtering)
        img = img.transform(img.size, self.perspectivestate.proj_type,
            perstate,
            filtering)
        arr = n.array(img)
        return arr

    def apply_perspective_rectim(self, rects, arr, affstate, perstate):
        rectarr = n.zeros(arr.shape)
        for i, rect in enumerate(rects):
            starti = max(0, rect[1])
            endi = min(rect[1]+rect[3], rectarr.shape[0])
            startj = max(0, rect[0])
            endj = min(rect[0]+rect[2], rectarr.shape[1])
            rectarr[starti:endi, startj:endj] = (i+1)*10
        rectarr = self.apply_perspective_arr(rectarr, affstate, perstate, filtering=Image.NONE)
        newrects = []
        for i, _ in enumerate(rects):
            try:
                newrects.append(pygame.Rect(self.get_bb(rectarr, eq=(i+1)*10)))
            except ValueError:
                pass
        return newrects

    def resize_rects(self, rects, arr, outheight):
        rectarr = n.zeros((arr.shape[0], arr.shape[1]))
        for i, rect in enumerate(rects):
            starti = max(0, rect[1])
            endi = min(rect[1]+rect[3], rectarr.shape[0])
            startj = max(0, rect[0])
            endj = min(rect[0]+rect[2], rectarr.shape[1])
            rectarr[starti:endi, startj:endj] = (i+1)*10
        rectarr = resize_image(rectarr, newh=outheight, filtering=Image.NONE)
        newrects = []
        for i, _ in enumerate(rects):
            try:
                newrects.append(pygame.Rect(self.get_bb(rectarr, eq=(i+1)*10)))
            except ValueError:
                pass
        return newrects

    def get_image(self):
        data = pygame.image.tostring(self.screen, 'RGBA')
        return n.array(Image.fromstring('RGBA', self.screen.get_size(), data))

    def get_ga_image(self, surf):
        r = pygame.surfarray.pixels_red(surf)
        a = pygame.surfarray.pixels_alpha(surf)
        r = r.reshape((r.shape[0], r.shape[1], 1))
        a = a.reshape(r.shape)
        return n.concatenate((r,a), axis=2).swapaxes(0,1)

    def arr_scroll(self, arr, dx, dy):
        arr = n.roll(arr, dx, axis=1)
        arr = n.roll(arr, dy, axis=0)
        return arr

    def get_bordershadow(self, bg_arr, colour):
        """
        Gets a border/shadow with the movement state [top, right, bottom, left].
        Inset or outset is random.
        """
        bs = self.borderstate.get_sample()
        outset = bs['outset']
        width = bs['width']
        position = bs['position']

        # make a copy
        border_arr = bg_arr.copy()
        # re-colour
        border_arr[...,0] = colour
        if outset:
            # dilate black (erode white)
            border_arr[...,1] = ndimage.grey_dilation(border_arr[...,1], size=(width, width))
            border_arr = self.arr_scroll(border_arr, position[0], position[1])

            # canvas = 255*n.ones(bg_arr.shape)
            # canvas = grey_blit(border_arr, canvas)
            # canvas = grey_blit(bg_arr, canvas)
            # pyplot.imshow(canvas[...,0], cmap=cm.Greys_r)
            # pyplot.show()

            return border_arr, bg_arr
        else:
            # erode black (dilate white)
            border_arr[...,1] = ndimage.grey_erosion(border_arr[...,1], size=(width, width))
            return bg_arr, border_arr

    def add_colour(self, canvas, fg_surf, border_surf=None):
        cs = self.colourstate.get_sample(2 + (border_surf is not None))
        # replace background
        pygame.PixelArray(canvas).replace((255,255,255), (cs[0],cs[0],cs[0]), distance=1.0)
        # replace foreground
        pygame.PixelArray(fg_surf).replace((0,0,0), (cs[1],cs[1],cs[1]), distance=0.99)

    def get_bb(self, arr, eq=None):
        if eq is None:
            v = n.nonzero(arr > 0)
        else:
            v = n.nonzero(arr == eq)
        xmin = v[1].min()
        xmax = v[1].max()
        ymin = v[0].min()
        ymax = v[0].max()
        return [xmin, ymin, xmax-xmin, ymax-ymin]

    def stack_arr(self, arrs):
        shp = list(arrs[0].shape)
        shp.append(1)
        tup = []
        for arr in arrs:
            tup.append(arr.reshape(shp))
        return n.concatenate(tup, axis=2)

    def imcrop(self, arr, rect):
        if arr.ndim > 2:
            return arr[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2],...]
        else:
            return arr[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

    def add_fillimage(self, arr):
        """
        Adds a fill image to the array.
        For blending this might be useful:
        - http://stackoverflow.com/questions/601776/what-do-the-blend-modes-in-pygame-mean
        - http://stackoverflow.com/questions/5605174/python-pil-function-to-divide-blend-two-images
        """


        fis = self.fillimstate.get_sample(arr)

        image = fis['image']
        blend_mode = fis['blend_mode']
        blend_amount = fis['blend_amount']
        blend_order = fis['blend_order']

        # change alpha of the image
        if blend_amount > 0:
            if blend_order:
                #image[...,1] *= blend_amount
                image[...,1] = (image[...,1]*blend_amount).astype(int)
                arr = grey_blit(image, arr, blend_mode=blend_mode)
            else:
                #arr[...,1] *= (1 - blend_amount)
                arr[...,1] = (arr[...,1]*(1-blend_amount)).astype(int)
                arr = grey_blit(arr, image, blend_mode=blend_mode)

        # pyplot.imshow(image[...,0], cmap=cm.Greys_r)
        # pyplot.show()

        return arr

    def mean_val(self, arr):
        return n.mean(arr[arr[...,1] > 0, 0].flatten())

    def surface_distortions(self, arr):
        ds = self.surfdiststate.get_sample()
        blur = ds['blur']

        origarr = arr.copy()
        arr = n.minimum(n.maximum(0, arr + n.random.normal(0, ds['noise'], arr.shape)), 255)
        # make some changes to the alpha
        arr[...,1] = ndimage.gaussian_filter(arr[...,1], ds['blur'])
        ds = self.surfdiststate.get_sample()
        arr[...,0] = ndimage.gaussian_filter(arr[...,0], ds['blur'])
        if ds['sharpen']:
            newarr_ = ndimage.gaussian_filter(origarr[...,0], blur/2)
            arr[...,0] = arr[...,0] + ds['sharpen_amount']*(arr[...,0] - newarr_)

        return arr

    def global_distortions(self, arr):
        # http://scipy-lectures.github.io/advanced/image_processing/#image-filtering
        ds = self.diststate.get_sample()

        blur = ds['blur']
        sharpen = ds['sharpen']
        sharpen_amount = ds['sharpen_amount']
        noise = ds['noise']

        newarr = n.minimum(n.maximum(0, arr + n.random.normal(0, noise, arr.shape)), 255)
        if blur > 0.1:
            newarr = ndimage.gaussian_filter(newarr, blur)
        if sharpen:
            newarr_ = ndimage.gaussian_filter(arr, blur/2)
            newarr = newarr + sharpen_amount*(newarr - newarr_)

        if ds['resample']:
            sh = newarr.shape[0]
            newarr = resize_image(newarr, newh=ds['resample_height'])
            newarr = resize_image(newarr, newh=sh)

        return newarr

    def get_rects_union_bb(self, rects, arr):
        rectarr = n.zeros((arr.shape[0], arr.shape[1]))
        for i, rect in enumerate(rects):
            starti = max(0, rect[1])
            endi = min(rect[1]+rect[3], rectarr.shape[0])
            startj = max(0, rect[0])
            endj = min(rect[0]+rect[2], rectarr.shape[1])
            rectarr[starti:endi, startj:endj] = 10
        return self.get_bb(rectarr)

    def apply_distortion_maps(self, arr, dispx, dispy):
        """
        Applies distortion maps generated from ElasticDistortionState
        """
        origarr = arr.copy()
        xx, yy = n.mgrid[0:dispx.shape[0], 0:dispx.shape[1]]
        xx = xx + dispx
        yy = yy + dispy
        coords = n.vstack([xx.flatten(), yy.flatten()])
        arr = ndimage.map_coordinates(origarr, coords, order=1, mode='nearest')
        return arr.reshape(origarr.shape)

    def generate_sample(self, display_text=None, display_text_length=None, outheight=None, pygame_display=False, random_crop=False, substring_crop=-1, char_annotations=False):
        """
        This generates the full text sample
        """

        if self.screen is None and pygame_display:
            self.screen = pygame.display.set_mode(self.sz)
            pygame.display.set_caption('WordRenderer')

        # clear bg
        # bg_surf = pygame.Surface(self.sz, SRCALPHA, 32)
        #bg_surf = bg_surf.convert_alpha()

        if display_text is None:
            # get the text to render
            display_text, label = self.corpus.get_sample(length=display_text_length)
        else:
            label = 0


        #print "generating sample for \"%s\"" % display_text

        # get a new font state


        fs = self.fontstate.get_sample()

        # clear bg
        # bg_surf = pygame.Surface(self.sz, SRCALPHA, 32)
        bg_surf = pygame.Surface((round(2.0 * fs['size'] * len(display_text)), self.sz[1]), SRCALPHA, 32)

        font = freetype.Font(fs['font'], size=fs['size'])

        # random params
        display_text = fs['capsmode'](display_text) if fs['random_caps'] else display_text
        font.underline = fs['underline']
        font.underline_adjustment = fs['underline_adjustment']
        font.strong = fs['strong']
        font.oblique = fs['oblique']
        font.strength = fs['strength']
        char_spacing = fs['char_spacing']


        font.antialiased = True
        font.origin = True

        #print 'hey inside generate_sample'

        # colour state
        cs = self.colourstate.get_sample(2 + fs['border'])
        #print cs

        #print 'hey inside generate_sample222'

        # baseline state



        mid_idx = int(math.floor(len(display_text)/2))
        curve = [0 for c in display_text]
        rotations = [0 for c in display_text]
        if fs['curved'] and len(display_text) > 1:
            bs = self.baselinestate.get_sample()
            for i, c in enumerate(display_text[mid_idx+1:]):
                curve[mid_idx+i+1] = bs['curve'](i+1)
                rotations[mid_idx+i+1] = -int(math.degrees(math.atan(bs['diff'](i+1)/float(fs['size']/2))))
            for i,c in enumerate(reversed(display_text[:mid_idx])):
                curve[mid_idx-i-1] = bs['curve'](-i-1)
                rotations[mid_idx-i-1] = -int(math.degrees(math.atan(bs['diff'](-i-1)/float(fs['size']/2))))
            mean_curve = sum(curve) / float(len(curve)-1)
            curve[mid_idx] = -1*mean_curve

        
        # render text (centered)
        char_bbs = []
        # place middle char
        rect = font.get_rect(display_text[mid_idx])
        rect.centerx = bg_surf.get_rect().centerx
        rect.centery = bg_surf.get_rect().centery + rect.height
        rect.centery +=  curve[mid_idx]
        bbrect = font.render_to(bg_surf, rect, display_text[mid_idx], rotation=rotations[mid_idx])

        bbrect.x = rect.x
        bbrect.y = rect.y - rect.height
        char_bbs.append(bbrect)



        # render chars to the right
        last_rect = rect
        for i, c in enumerate(display_text[mid_idx+1:]):
            char_fact = 1.0
            if fs['random_kerning']:
                char_fact += fs['random_kerning_amount']*n.random.randn()
            newrect = font.get_rect(c)
            newrect.y = last_rect.y
            newrect.topleft = (last_rect.topright[0] + char_spacing*char_fact, newrect.topleft[1])
            newrect.centery = max(0 + newrect.height*1, min(self.sz[1] - newrect.height*1, newrect.centery + curve[mid_idx+i+1]))
            try:
                bbrect = font.render_to(bg_surf, newrect, c, rotation=rotations[mid_idx+i+1])
            except ValueError:
                bbrect = font.render_to(bg_surf, newrect, c)
            bbrect.x = newrect.x
            bbrect.y = newrect.y - newrect.height
            char_bbs.append(bbrect)
            last_rect = newrect
        # render chars to the left
        last_rect = rect
        for i, c in enumerate(reversed(display_text[:mid_idx])):
            char_fact = 1.0
            if fs['random_kerning']:
                char_fact += fs['random_kerning_amount']*n.random.randn()
            newrect = font.get_rect(c)
            newrect.y = last_rect.y
            newrect.topright = (last_rect.topleft[0] - char_spacing*char_fact, newrect.topleft[1])
            newrect.centery = max(0 + newrect.height*1, min(self.sz[1] - newrect.height*1, newrect.centery + curve[mid_idx-i-1]))
            try:
                bbrect = font.render_to(bg_surf, newrect, c, rotation=rotations[mid_idx-i-1])
            except ValueError:
                bbrect = font.render_to(bg_surf, newrect, c)
            bbrect.x = newrect.x
            bbrect.y = newrect.y - newrect.height
            char_bbs.append(bbrect)
            last_rect = newrect


        #show
        # self.screen = pygame.display.set_mode(bg_surf.get_size())
        # self.screen.fill((255,255,255))
        # self.screen.blit(bg_surf, (0,0))
        # # for bb in char_bbs:
        # #     pygame.draw.rect(self.screen, (255,0,0), bb, 2)
        # pygame.display.flip()
        # save_screen_img(self.screen, '/Users/jaderberg/Desktop/0.jpg')
        # wait_key()


        bg_arr = self.get_ga_image(bg_surf)

        # colour text
        bg_arr[...,0] = cs[0]

        # # do elastic distortion described by http://research.microsoft.com/pubs/68920/icdar03.pdf
        # dispx, dispy = self.elasticstate.sample_transformation(bg_arr[...,0].shape)
        # bg_arr[...,1] = self.apply_distortion_maps(bg_arr[...,1], dispx, dispy)
        #
        # # show
        # self.screen = pygame.display.set_mode(bg_surf.get_size())
        # canvas = 255*n.ones(bg_arr.shape)
        # globalcanvas = grey_blit(bg_arr, canvas)[...,0]
        # rgb_canvas = self.stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # self.screen.blit(pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1)), (0,0))
        # pygame.display.flip()
        # wait_key()

        # border/shadow
        if fs['border']:
            l1_arr, l2_arr = self.get_bordershadow(bg_arr, cs[2])
        else:
            l1_arr = bg_arr


        # show individiual layers (fore, bord, back)
        # canvas = (255*n.ones(l1_arr.shape)).astype(l1_arr.dtype)
        # globalcanvas = grey_blit(l2_arr, canvas)[...,0]
        # rgb_canvas = self.stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # self.screen.blit(pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1)), (0,0))
        # pygame.display.flip()
        # save_screen_img(self.screen, '/Users/jaderberg/Desktop/1.jpg')
        # wait_key()
        # canvas = (255*n.ones(l1_arr.shape)).astype(l1_arr.dtype)
        # globalcanvas = grey_blit(l1_arr, canvas)[...,0]
        # rgb_canvas = self.stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # self.screen.blit(pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1)), (0,0))
        # pygame.display.flip()
        # save_screen_img(self.screen, '/Users/jaderberg/Desktop/2.jpg')
        # wait_key()
        # self.screen.fill((cs[1],cs[1],cs[1]))
        # pygame.display.flip()
        # save_screen_img(self.screen, '/Users/jaderberg/Desktop/3.jpg')
        # wait_key()

        # show
        # canvas = (255*n.ones(l1_arr.shape)).astype(l1_arr.dtype)
        # canvas[...,0] = cs[1]
        # globalcanvas = grey_blit(l1_arr, canvas)
        # if fs['border']:
        #     globalcanvas = grey_blit(l2_arr, globalcanvas)
        # globalcanvas = globalcanvas[...,0]
        # rgb_canvas = self.stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # canvas_surf = pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1))
        # # for char_bb in char_bbs:
        # #     pygame.draw.rect(canvas_surf, (255,0,0), char_bb, 2)
        # self.screen = pygame.display.set_mode(canvas_surf.get_size())
        # self.screen.blit(canvas_surf, (0, 0))
        # pygame.display.flip()
        # save_screen_img(self.screen, '/Users/jaderberg/Desktop/4.jpg')
        # wait_key()

        # do rotation and perspective distortion
        affstate = self.affinestate.sample_transformation(l1_arr.shape)
        perstate = self.perspectivestate.sample_transformation(l1_arr.shape)
        l1_arr[...,1] = self.apply_perspective_arr(l1_arr[...,1], affstate, perstate)
        if fs['border']:
            l2_arr[...,1] = self.apply_perspective_arr(l2_arr[...,1], affstate, perstate)
        if char_annotations:
            char_bbs = self.apply_perspective_rectim(char_bbs, l1_arr[...,1], affstate, perstate)
            # order char_bbs by left to right
            xvals = [bb.x for bb in char_bbs]
            idx = [i[0] for i in sorted(enumerate(xvals), key=lambda x:x[1])]
            char_bbs = [char_bbs[i] for i in idx]


        if n.random.rand() < substring_crop and len(display_text) > 4 and char_annotations:
            # randomly crop to just a sub-string of the word
            start = n.random.randint(0, len(display_text)-1)
            stop = n.random.randint(min(start+1,len(display_text)), len(display_text))
            display_text = display_text[start:stop]
            char_bbs = char_bbs[start:stop]
            # get new bb of image
            bb = pygame.Rect(self.get_rects_union_bb(char_bbs, l1_arr))
        else:
            # get bb of text
            if fs['border']:
                bb = pygame.Rect(self.get_bb(grey_blit(l2_arr, l1_arr)[...,1]))
            else:
                bb = pygame.Rect(self.get_bb(l1_arr[...,1]))
        if random_crop:
            bb.inflate_ip(10*n.random.randn()+15, 10*n.random.randn()+15)
        else:
            inflate_amount = int(0.4*bb[3])
            bb.inflate_ip(inflate_amount, inflate_amount)


        # crop image
        l1_arr = self.imcrop(l1_arr, bb)
        if fs['border']:
            l2_arr = self.imcrop(l2_arr, bb)
        if char_annotations:
            # adjust char bbs
            for char_bb in char_bbs:
                char_bb.move_ip(-bb.x, -bb.y)
        canvas = (255*n.ones(l1_arr.shape)).astype(l1_arr.dtype)
        canvas[...,0] = cs[1]



        # show
        # globalcanvas = grey_blit(l1_arr, canvas)
        # if fs['border']:
        #     globalcanvas = grey_blit(l2_arr, globalcanvas)
        # globalcanvas = globalcanvas[...,0]
        # rgb_canvas = self.stack_arr((globalcanvas, globalcanvas, globalcanvas))
        # canvas_surf = pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1))
        # # for char_bb in char_bbs:
        # #     pygame.draw.rect(canvas_surf, (255,0,0), char_bb, 2)
        # self.screen = pygame.display.set_mode(canvas_surf.get_size())
        # self.screen.blit(canvas_surf, (0, 0))
        # pygame.display.flip()
        # save_screen_img(self.screen, '/Users/jaderberg/Desktop/5.jpg')
        # wait_key()


        # add in natural images
        try:
            canvas = self.add_fillimage(canvas)
            l1_arr = self.add_fillimage(l1_arr)
            if fs['border']:
                l2_arr = self.add_fillimage(l2_arr)
        except Exception:
            print "\tfillimage error"
            return None


        # add per-surface distortions
        l1_arr = self.surface_distortions(l1_arr)
        if fs['border']:
            l2_arr = self.surface_distortions(l2_arr)



        # compose global image
        blend_modes = [MJBLEND_NORMAL, MJBLEND_ADD, MJBLEND_MULTINV, MJBLEND_SCREEN, MJBLEND_MAX]
        count = 0
        while True:
            globalcanvas = grey_blit(l1_arr, canvas, blend_mode=random.choice(blend_modes))
            if fs['border']:
                globalcanvas = grey_blit(l2_arr, globalcanvas, blend_mode=random.choice(blend_modes))
            globalcanvas = globalcanvas[...,0]
            std = n.std(globalcanvas.flatten())
            count += 1
            #print count
            if std > 20:
                break
            if count > 10:
                print "\tcan't get good contrast"
                return None
            	
        canvas = globalcanvas

        # do elastic distortion described by http://research.microsoft.com/pubs/68920/icdar03.pdf
        # dispx, dispy = self.elasticstate.sample_transformation(canvas.shape)
        # canvas = self.apply_distortion_maps(canvas, dispx, dispy)

        # add global distortions
        canvas = self.global_distortions(canvas)

        # noise removal
        canvas = ndimage.filters.median_filter(canvas, size=(3,3))

        # resize
        if outheight is not None:
            if char_annotations:
                char_bbs = self.resize_rects(char_bbs, canvas, outheight)
            canvas = resize_image(canvas, newh=outheight)




        # FINISHED, SHOW ME SOMETHING
        if pygame_display:
            rgb_canvas = self.stack_arr((canvas, canvas, canvas))
            canvas_surf = pygame.surfarray.make_surface(rgb_canvas.swapaxes(0,1))
            # for char_bb in char_bbs:
            #     pygame.draw.rect(canvas_surf, (255,0,0), char_bb, 2)
            self.screen = pygame.display.set_mode(canvas_surf.get_size())
            self.screen.blit(canvas_surf, (0, 0))
            pygame.display.flip()


        #pyplot.imshow(self.get_image())
        #pyplot.show()

        # print char_bbs[0]

        return {
            'image': canvas,
            'text': display_text,
            'label': label,
            'chars': n.array([[c.x, c.y, c.width, c.height] for c in char_bbs])
        }



if __name__ == "__main__":

    fillimstate = SVTFillImageState("/Users/jaderberg/Data/TextSpotting/DataDump/svt1", "/Users/jaderberg/Data/TextSpotting/DataDump/svt1/SVT-train.mat")
    fs = FontState()
    # fs.border = 1.0

    # corpus = SVTCorpus({'unk_probability': 0.5})
    corpus = RandomCorpus({'min_length': 1, 'max_length': 10})
    WR = WordRenderer(fontstate=fs, fillimstate=fillimstate, colourstate=TrainingCharsColourState, corpus=corpus)
    while True:
        data = WR.generate_sample(pygame_display=True, substring_crop=0, random_crop=True, char_annotations=True)
        if data is not None:
            print data['text']
        # save_screen_img(WR.screen, '/Users/jaderberg/Desktop/6.jpg', 70)
        wait_key()

    # WR = WordRenderer(fontstate=fs, fillimstate=fillimstate, corpus=Corpus, colourstate=TrainingCharsColourState)
    # towrite = "Worcester College Rocks".split()
    # perrow = 4.0
    # rows = math.ceil(len(towrite)/perrow)
    # cols = int(perrow)
    # num = len(towrite)
    # for i, w in enumerate(towrite):
    #     pyplot.subplot(rows, cols, i+1)
    #     pyplot.imshow(WR.generate_sample(display_text=w, outheight=32)['image'], cmap=cm.Greys_r)
    #     pyplot.axis('off')
    # pyplot.tight_layout()
    # pyplot.show()
