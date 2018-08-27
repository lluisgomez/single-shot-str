import cv2
import sys
import numpy as np

def import_cphoc():
    import ctypes
    lib_c = ctypes.CDLL('./cphoc/cphoc.so')
    lib_c.build_phoc.restype = ctypes.py_object
    lib_c.build_phoc.argtypes = [ctypes.c_char_p]
    return lib_c.build_phoc

def img_preprocess(im, shape=(608, 608, 3), letterbox=False):
    if type(im) is not np.ndarray:
        im = cv2.imread(im)

    if letterbox:
        top_border, bottom_border, left_border, right_border = (0,0,0,0)
        h,w,c = im.shape
        if w>h:
          letterbox = cv2.resize(im, (shape[1], int(float(h)*(float(shape[1])/w))))
          top_border    = int( float(shape[0] - int(float(h)*(float(shape[1])/w))) / 2. )
          bottom_border = top_border
          if top_border+bottom_border+int(float(h)*(float(shape[1])/w)) < shape[0]:
            bottom_border +=1
        else:
          letterbox = cv2.resize(im, (int(float(w)*(float(shape[0])/h)), shape[0]))
          left_border  = int( float(shape[1] - int(float(w)*(float(shape[0])/h))) / 2. )
          right_border = left_border
          if left_border+right_border+int(float(w)*(float(shape[0])/h)) < shape[1]:
            right_border +=1
        im = cv2.copyMakeBorder(letterbox, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT, value=[127,127,127])
    else:
        im = cv2.resize(im, (shape[1], shape[0]))

    im = im / 255.
    im = im[:,:,::-1]

    return im

def expit(x):
    return 1. / (1. + np.exp(-x))


class tcolors:
    INFO    = '\033[94m'
    ERROR   = '\033[91m'
    OK      = '\033[92m'
    WARNING = '\033[93m'
    ENDC    = '\033[0m'

def print_info(msg):
    sys.stdout.write(tcolors.INFO+msg+tcolors.ENDC)
    sys.stdout.flush()

def print_err(msg):
    sys.stdout.write(tcolors.ERROR+msg+tcolors.ENDC)
    sys.stdout.flush()

def print_ok(msg):
    sys.stdout.write(tcolors.OK+msg+tcolors.ENDC)
    sys.stdout.flush()

def print_warn(msg):
    sys.stdout.write(tcolors.WARNING+msg+tcolors.ENDC)
    sys.stdout.flush()

def print_progress(progress, msg=''):
    sys.stdout.write('\r')
    sys.stdout.write(msg+" [%-20s] %d%%" % ('='*int(progress/5), progress))
    sys.stdout.flush()
