from __future__ import print_function
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
import argparse
import json
from json import encoder
import os

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set up file names and pathes
dataDir='.'
dataType='val2014'
algName = 'fakecap'
annFile='%s/annotations/captions_%s.json'%(dataDir,dataType)
subtypes=['results', 'evalImgs', 'eval']
[resFile, evalImgsFile, evalFile]= \
['%s/results/captions_%s_%s_%s.json'%(dataDir,dataType,algName,subtype) for subtype in subtypes]

parser = argparse.ArgumentParser(description="Controllable dictionary example generation.")
parser.add_argument('--testfile', type=str, default='', help='the path of the generated file.')
parser.add_argument('--gtfile', type=str, default='', help='the path of the gt file.')
args = parser.parse_args()

if os.path.isdir(args.testfile):
    paths = [os.path.join(args.testfile, f) for f in os.listdir(args.testfile)]
else:
    paths = [args.testfile]

for path in paths:
    print("=======================================")
    print(f"evaluate {path} ...")
    print("=======================================")
    # create coco object and cocoRes object
    coco = COCO(args.gtfile) # gt
    cocoRes = coco.loadRes(path) # generated

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()

    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        print('%s: %.3f'%(metric, score))
        
