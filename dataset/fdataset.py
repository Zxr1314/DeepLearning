'''
File dataset
Read data when batching
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np

from dataset.dataset import DataSet

class FDataSet(DataSet):
    def __init__(self, common_params, dataset_params):
        '''
        Initialization
        :param common_params:
        :param dataset_params:
        '''
        self.batch_size = common_params['batch_size']
        self.dimension = common_params['dimension']
        self.dtype = dataset_params['dtype']
        self.bSize = False
        self.channel = common_params['channel']
        if 'width' in common_params and 'height' in common_params:
            self.width = common_params['width']
            self.height = common_params['height']
            if self.dimension==3:
                self.depth = common_params['depth']
            self.bSize = True
        else:
            if 'size_path' in dataset_params:
                self.size_path = dataset_params['size_path']
                self.size_files = self.__get_files(self.size_path)
                self.size_files.sort()
                self.batch_size = 1
            else:
                raise Exception('Image sizes unknown!')
        if 'random' in dataset_params:
            self.random = dataset_params['random']
        else:
            self.random = True
        self.data_path = dataset_params['data_path']
        self.label_path = dataset_params['label_path']
        self.data_files = self.__get_files(self.data_path)
        self.label_files = self.__get_files(self.label_path)
        self.data_files.sort()
        self.label_files.sort()

        if not self.random:
            self.data_files_batch = []
            self.label_files_batch = []
            i = 0
            while i < len(self.data_files):
                self.data_files_batch.append(self.data_files[i:i+self.batch_size])
                self.label_files_batch.append(self.label_files[i:i+self.batch_size])
                i += self.batch_size
            self.n_batch = len(self.data_files_batch)
            self.i_batch = 0

        self.testing = common_params['testing']
        if self.testing:
            if 'test_batch_size' not in common_params or 'test_data_path' not in dataset_params or 'test_label_path' not in dataset_params:
                raise Exception('Testing chosen while parameters not given!')
            self.test_batch_size = common_params['test_batch_size']
            self.test_data_path = dataset_params['test_data_path']
            self.test_label_path = dataset_params['test_label_path']
            self.test_data_files = self.__get_files(self.test_data_path)
            self.test_label_files = self.__get_files(self.test_label_path)
            self.test_data_files.sort()
            self.test_label_files.sort()
            if not self.bSize:
                if 'test_size_path' in dataset_params:
                    self.test_size_path = dataset_params['test_size_path']
                    self.test_size_files = self.__get_files(self.test_size_path)
                    self.test_size_files.sort()
                    self.test_batch_size = 1
            self.test_data_files_batch = []
            self.test_label_files_batch = []
            i = 0
            while i < len(self.test_data_files):
                self.test_data_files_batch.append(self.test_data_files[i:i+self.test_batch_size])
                self.test_label_files_batch.append(self.test_label_files[i:i+self.test_batch_size])
                i += self.test_batch_size
            self.test_n_batch = len(self.test_data_files_batch)
            self.test_i_batch = 0
        return

    def __get_files(self, path):
        '''
        Get files from path
        :param path:
        :return:
        '''
        Files = []
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                if '.dat' in filename.lower():
                    Files.append(os.path.join(dirName, filename))
        return Files

    def batch(self):
        if self.random:
            data, label = self.__random_batch(self.batch_size, self.data_files, self.label_files)
        else:
            if self.bSize:
                data, label, self.i_batch = self.__sequential_batch(self.i_batch, self.n_batch, self.batch_size, self.data_files_batch, self.label_files_batch)
            else:
                data, label, self.i_batch = self.__sequential_batch(self.i_batch, self.n_batch, self.batch_size, self.data_files, self.label_files, self.size_files)
        return data, label

    def test_batch(self):
        '''

        :return:
        '''
        if self.bSize:
            tdata, tlabel, self.test_i_batch = self.__sequential_batch(self.test_i_batch, self.test_n_batch, self.test_batch_size, self.test_data_files_batch, self.test_label_files_batch)
        else:
            tdata, tlabel, self.test_i_batch = self.__sequential_batch(self.test_i_batch, self.test_n_batch, self.test_batch_size, self.test_data_files,
                                                    self.test_label_files, self.test_size_files)
        return tdata, tlabel

    def test_random_batch(self):
        '''

        :return:
        '''
        if self.bSize:
            tdata, tlabel = self.__random_batch(self.test_batch_size, self.test_data_files, self.test_label_files)
        else:
            tdata, tlabel = self.__random_batch(self.test_batch_size, self.test_data_files, self.test_label_files, self.test_size_files)
        return tdata, tlabel

    def get_n_test_batch(self):
        '''

        :return:
        '''
        return self.test_n_batch

    def __random_batch(self, batch_size, data_files, label_files, size_files=None):
        '''

        :return:
        '''
        if not self.bSize:
            index = random.randint(0, len(data_files))
            data = np.fromfile(data_files[index], dtype=self.dtype).astype(np.float32)
            label = np.fromfile(label_files[index], dtype=self.dtype).astype(np.float32)
            size = np.loadtxt(size_files[index], dtype=np.uint16)
            if self.dimension==2:
                data.shape = [1, size[0], size[1], self.channel]
                label.shape = [1, size[0], size[1], 1]
            else:
                data.shape = [1, size[0], size[1], size[2], self.channel]
                label.shape = [1, size[0], size[1], size[2], 1]
        else:
            if self.dimension==2:
                data = np.zeros([self.batch_size, self.height, self.width, self.channel], dtype=np.float32)
                label = np.zeros([self.batch_size, self.height, self.width, 1], dtype=np.float32)
            else:
                data = np.zeros([self.batch_size, self.depth, self.height, self.width, self.channel], dtype=np.float32)
                label = np.zeros([self.batch_size, self.depth, self.height, self.width, 1], dtype=np.float32)
            index = random.sample(range(len(data_files)), self.batch_size)
            for i in xrange(self.batch_size):
                d = np.fromfile(data_files[index[i]], dtype=self.dtype).astype(np.float32)
                l = np.fromfile(label_files[index[i]], dtype=self.dtype).astype(np.float32)
                if self.dimension==2:
                    d.shape = [1, self.height, self.width, self.channel]
                    l.shape = [1, self.height, self.width, 1]
                else:
                    d.shape = [1, self.depth, self.height, self.width, self.channel]
                    l.shape = [1, self.depth, self.height, self.width, 1]
                data[i,:] = d
                label[i,:] = l
        return data, label

    def __sequential_batch(self, i_batch, n_batch, batch_size, data_files_batch, label_files_batch, size_files_batch=None):
        '''

        :return:
        '''
        if not self.bSize:
            index = random.randint(0, len(self.data_files))
            data = np.fromfile(data_files_batch[i_batch], dtype=self.dtype).astype(np.float32)
            label = np.fromfile(label_files_batch[i_batch], dtype=self.dtype).astype(np.float32)
            size = np.loadtxt(size_files_batch[i_batch], dtype=np.uint16)
            if self.dimension==2:
                data.shape = [1, size[0], size[1], self.channel]
                label.shape = [1, size[0], size[1], 1]
            else:
                data.shape = [1, size[0], size[1], size[2], self.channel]
                label.shape = [1, size[0], size[1], size[2], 1]
        else:
            if self.dimension==2:
                data = np.zeros([batch_size, self.height, self.width, self.channel], dtype=np.float32)
                label = np.zeros([batch_size, self.height, self.width, 1], dtype=np.float32)
            else:
                data = np.zeros([batch_size, self.depth, self.height, self.width, self.channel], dtype=np.float32)
                label = np.zeros([batch_size, self.depth, self.height, self.width, 1], dtype=np.float32)
            for i in xrange(len(data_files_batch[i_batch])):
                d = np.fromfile(data_files_batch[i_batch][i], dtype=self.dtype).astype(np.float32)
                l = np.fromfile(label_files_batch[i_batch][i], dtype=self.dtype).astype(np.float32)
                if self.dimension==2:
                    d.shape = [1, self.height, self.width, self.channel]
                    l.shape = [1, self.height, self.width, 1]
                else:
                    d.shape = [1, self.depth, self.height, self.width, self.channel]
                    l.shape = [1, self.depth, self.height, self.width, 1]
                data[i,:] = d
                label[i,:] = l
        i_batch = (i_batch+1)%n_batch
        return data, label, i_batch