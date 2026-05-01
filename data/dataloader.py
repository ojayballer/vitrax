import os
import glob
import numpy as np
from PIL import Image


def get_files(data_dir,split='train'):
    ##get all file paths and labels for a split
    if split == 'train':
        root = data_dir+'/train'
        classes = sorted(os.listdir(root))
        cmap = {c:i for i,c in enumerate(classes)}

        files = []
        lbls = []
        for c in classes:
            imgs = glob.glob(root+'/'+c+'/images/*.JPEG')
            for p in imgs:
                files.append(p)
                lbls.append(cmap[c])

    else:
        root = data_dir+'/val'
        classes = sorted(os.listdir(data_dir+'/train'))
        cmap = {c:i for i,c in enumerate(classes)}

        files = []
        lbls = []
        with open(root+'/val_annotations.txt') as f:
            for line in f:
                parts = line.strip().split('\t')
                files.append(root+'/images/'+parts[0])
                lbls.append(cmap[parts[1]])

    ##shuffle
    idx = np.random.permutation(len(files))
    files = [files[i] for i in idx]
    lbls = [lbls[i] for i in idx]
    return files,lbls


def load_batch(files,lbls):
    ##load a batch of images from file paths
    imgs = []
    for f in files:
        img = np.array(Image.open(f).convert('RGB'),dtype=np.float32)/255.0
        imgs.append(img)
    return np.array(imgs,dtype=np.float32),np.array(lbls,dtype=np.int32)
