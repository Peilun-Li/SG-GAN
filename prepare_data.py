import sys
import os
import argparse
from glob import glob
from random import shuffle
import shutil

def prepare(img_dir, seg_dir, img_target_dir, seg_target_dir, replace_names=None):
  imgs = set(glob(img_dir+"*.png"))
  segs = set(glob(seg_dir+"*.png"))
  pairs = []
  for img_path in list(imgs):
    seg_path = seg_dir + (img_path.split("/")[-1].replace(replace_names[0], replace_names[1]) if replace_names else img_path.split("/")[-1])
    if seg_path in segs:
      pairs.append((img_path, seg_path))
  print "candidates:", len(pairs)
  if len(pairs) < args["train_size"] + args["test_size"]:
    print "candidates not enough!"
    return
  if not os.path.exists(img_target_dir):
    os.makedirs(img_target_dir)
    os.makedirs(img_target_dir.replace("train", "test"))
  if not os.path.exists(seg_target_dir):
    os.makedirs(seg_target_dir)
    os.makedirs(seg_target_dir.replace("train", "test"))
  shuffle(pairs)
  for i in range(args["train_size"]):
    shutil.copy2(pairs[i][0], img_target_dir + (pairs[i][0].split("/")[-1].replace(replace_names[0], "") if replace_names else ""))
    shutil.copy2(pairs[i][1], seg_target_dir + (pairs[i][1].split("/")[-1].replace(replace_names[1], "") if replace_names else ""))
  for i in range(args["train_size"], args["train_size"]+args["test_size"]):
    shutil.copy2(pairs[i][0], img_target_dir.replace("train", "test") + (pairs[i][0].split("/")[-1].replace(replace_names[0], "") if replace_names else ""))
    shutil.copy2(pairs[i][1], seg_target_dir.replace("train", "test") + (pairs[i][1].split("/")[-1].replace(replace_names[1], "") if replace_names else ""))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--A_imagepath", "-Ai", type=str, default="/home/lpl/data/playing/images/", help="dataset A's image path")
  parser.add_argument("--A_segpath", "-As", type=str, default="/home/lpl/data/playing/labels/", help="dataset A's segmentation path")
  # cp `find train/ -name "*.png"` all_train/
  parser.add_argument("--B_imagepath", "-Bi", type=str, default="/home/lpl/data/cityscape/leftImg8bit/all_train/", help="dataset B's image path")
  parser.add_argument("--B_segpath", "-Bs", type=str, default="/home/lpl/data/cityscape/gtFine/all_train/", help="dataset B's segmentation path")
  parser.add_argument("--train_size", "-tr", type=int, default=2000, help="number of training examples for each dataset")
  parser.add_argument("--test_size", "-te", type=int, default=500, help="number of test examples for each dataset")
  args = vars(parser.parse_args())

  prepare(args["A_imagepath"], args["A_segpath"], "./datasets/gta/trainA/", "./datasets/gta/trainA_seg/")
  prepare(args["B_imagepath"], args["B_segpath"], "./datasets/gta/trainB/", "./datasets/gta/trainB_seg/", replace_names=("_leftImg8bit", "_gtFine_color"))

