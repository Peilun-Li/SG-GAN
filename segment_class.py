from collections import defaultdict
import os
import sys
from glob import glob
from scipy.misc import imread, imsave, toimage
import numpy as np
from multiprocessing import Pool as ProcessPool
num_seg_masks = 8
# vehicles: 1
# pedestrians: 2
# cyclist: 3
# roads: 4
# buildings: 5
# sky: 6
# tree: 7
# others: 0


# https://bitbucket.org/visinf/projects-2016-playing-for-data/src/6afee1a5923f452e741c9256f5fb78f2b3882ee2/label/initLabels.m?at=master&fileviewer=file-view-default
"""
('0,0,0', 'unlabeled')
('0,0,0', 'ego vehicle')
('0,0,0', 'rectification border')
('0,0,0', 'out of roi')
('20,20,20', 'static')
('111,74,0', 'dynamic')
('81,0,81', 'ground')
('128,64,128', 'road')
('244,35,232', 'sidewalk')
('250,170,160', 'parking')
('230,150,140', 'rail track')
('70,70,70', 'building')
('102,102,156', 'wall')
('190,153,153', 'fence')
('180,165,180', 'guard rail')
('150,100,100', 'bridge')
('150,120,90', 'tunnel')
('153,153,153', 'pole')
('153,153,153', 'polegroup')
('250,170,30', 'traffic light')
('220,220,0', 'traffic sign')
('107,142,35', 'vegetation')
('152,251,152', 'terrain')
('70,130,180', 'sky')
('220,20,60', 'person')
('255,0,0', 'rider')
('0,0,142', 'car')
('0,0,70', 'truck')
('0,60,100', 'bus')
('0,0,90', 'caravan')
('0,0,110', 'trailer')
('0,80,100', 'train')
('0,0,230', 'motorcycle')
('119,11,32', 'bicycle')
('0,0,142', 'license plate')
"""
# https://github.molgen.mpg.de/mohomran/cityscapes/blob/master/scripts/helpers/labels.py
def cityscape():
  rgb_to_maskidx = defaultdict(int)
  maps = [((128,64,128),4), ((244,35,232),4), ((250,170,160),4), ((230,150,140),4), ((70,70,70),5)
         , ((102,102,156),5), ((190,153,153),5), ((180,165,180),5), ((150,100,100),5), ((150,120,90),5)
         , ((107,142,35),7), ((70,130,180),6), ((220,20,60),2), ((255,0,0),2), ((0,0,142),1), ((0,0,70),1)
         , ((0,60,100),1), ((0,0,90),1), ((0,0,110),1), ((0,0,230),3), ((119,11,32),3)]
  for k,v in maps:
    rgb_to_maskidx[k] = v
  return rgb_to_maskidx 

def A_maskmap():
  return cityscape()

def B_maskmap():
  return cityscape()

def preprocess_master(src):
  dst = src.replace("_seg", "_seg_class")
  if not os.path.exists(dst):
    os.makedirs(dst)
  segs = set(glob(src+"/*.png"))
  pool = ProcessPool(8)
  pool.map(preprocess, segs)
 
def preprocess(image_seg):
  base_name = os.path.basename(image_seg)
  print "processing", base_name
  img = imread(image_seg)
  M, N = img.shape[:2]
  seg_class = np.zeros((M, N)).astype(np.int)
  for x in range(M):
    for y in range(N):
      seg_class[x,y] = maskmap[tuple(img[x,y,:3])]
  toimage(seg_class, cmin=0, cmax=255).save(image_seg.replace("_seg", "_seg_class")) 

if __name__ == "__main__":
  maskmap = A_maskmap()
  preprocess_master("datasets/gta/trainA_seg")
  preprocess_master("datasets/gta/trainB_seg")
