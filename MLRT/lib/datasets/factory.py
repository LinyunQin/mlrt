# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.cityscape import cityscape
from datasets.dgunionlable import dgunionlable

from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.vg import vg

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ['2007', '2012']:
  for split in ['train_s','train_s2','train_s3', 'train_t', 'train_val', 'test_s', 'test_t','test_all']:
    name = 'cityscape_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: cityscape(split, year))

for year in ['2007', '2012']:
    for testdataset in ['unionvoc', 'unioncityscape', 'kitti', 'watercolor', 'clipart', 'sim10k','kitti','bdd100k','fogycityscape','raincityscape']:
        for dataset in ['unionvoc', 'unioncityscape', 'kitti', 'watercolor', 'clipart', 'sim10k','kitti','bdd100k','fogycityscape','raincityscape']:
            for source in ['_s1','_s2','_s3','_s4','_s5','_s6','_s7','_s8','_s10','_s14','_s15','_s16','_s17','_s18','_s19','_single1','_single2','_single3','_single4','_single5','_single6','_s30','_s31','_s32','_t','_factkitti','_factsim10k','_fsdrkitti','_fsdrsim10k','_fsdrkittiv2','_fsdrsim10kv2']:

                for split in ['train','test','trainval','train_h','mval','mtrain']:
                    name = '{}_{}_{}'.format(dataset+source, year, split+testdataset)
                    __sets[name] = (lambda testdataset=testdataset, dataset=dataset, split=split, year=year, source=source: dgunionlable(testdataset, dataset, source, split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
  for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))
        
# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

def get_imdb(name):
  #print(name)
  #print(list_imdbs())
  #print(__sets['kitti_2007_train'])
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
