"""
Batch Loader by Donny You
"""

import cv2
import numpy as np
import numpy.random as nr
from random import shuffle
import os

class BatchLoader(object):

    def __init__(self, file_path, batch_size):
        self.batch_size = batch_size
        self.labels, self.im_list = self.image_dir_processor(file_path)
        self.idx = 0
        self.data_num = len(self.labels)
        self.rnd_list = np.arange(self.data_num)
        shuffle(self.rnd_list)

    def next_batch(self):
        batch_images = []
        batch_labels = []

        for i in xrange (self.batch_size):
            if self.idx != self.data_num:
                cur_idx = self.rnd_list[self.idx]
                im_path = self.im_list[cur_idx]
                image = cv2.imread(im_path)
                batch_images.append(image)
                batch_labels.append(self.labels[cur_idx])

                self.idx +=1
            else:
                self.idx = 0
                shuffle(self.rnd_list)

        batch_images = np.array(batch_images).astype(np.float32)
        batch_labels = np.array(batch_labels).astype(np.float32)
        return batch_images, batch_labels 
        
    def image_dir_processor(self, file_path):
        labels = []
        im_path_list = []
        if not os.path.exists(file_path):
            print "File %s not exists." % file_path
            exit()

        with open(file_path, "r") as fr:
            for line in fr.readlines():
                terms = line.rstrip().split()
                label = int(terms[1])
                im_path_list.append(terms[0])
                labels.append(label)

        return labels, im_path_list
