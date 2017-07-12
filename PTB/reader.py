# -*- coding: UTF-8 -*-
"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import pickle

import tensorflow as tf


def _read_words(filename):
    #从文件读取数据，以空格为切分符
    ###########################
    # Args:
    #   filename    - 文件完整路径

    # Returns:
    #   list()
    #   每个元素类型为unicode
    if not os.path.exists(filename):
        raise ValueError("_read_words : file not exist -",filename)
    with tf.gfile.GFile(filename, "r") as f:
        if sys.version_info[0] >= 3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
    # Args:
    #   filename    - 文件完整路径

    # Returns:
    #   dict()
    #   key:单词 value:序号
    data = _read_words(filename)

    #为可哈希对象计数，字典的子类
    counter = collections.Counter(data)
    #词频降序列表，list of tuple(itm, count)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    #          list([tuple(itm),tuple(count)])
    words, _ = list(zip(*count_pairs))
    #从word到（唯一）int的dict
    word_to_id = dict(zip(words, range(len(words))))
    print("A vocab_dict is generated holding",len(words), "words")
    return word_to_id

def _save_vocab(word_to_id,filename):
    # Args:
    #   word_to_id      - 字典
    #   filename        - 路径 
    with open(filename,'w')  as fhand:
        pickle.dump(word_to_id,fhand)  


def _load_vocab(filename):
    # Args:
    #   filename    - 文件完整路径

    # Returns:
    #   dict()
    #   key:单词 value:序号
    with open(filename,'r') as fhand:
        word_to_id = pickle.load(fhand) 
    return word_to_id    


def _file_to_word_ids(filename, word_to_id):
    # Args:
    #   filename    - 文件完整路径

    # Returns:
    #   list of integer
    #   每个整形数为单词对应的唯一编号
    data = _read_words(filename)
    top = len(word_to_id)
    #舍弃未出现在字典中的单词
    return [word_to_id[word] if word in word_to_id else top for word in data ]


def ptb_raw_data(data_path=None):
    # 从路径读取文本文件
    # 以空格为分隔符将字符串转化为整形ids

    # Args:
    #   data_path: 训练/验证/测试 文本所在路径

    # Returns:
    #   tuple (train_data, valid_data, test_data, vocabulary)
    #   PTBIterator 所需的数据实例
    dict_path = os.path.join("..", "model", "dict.pickle")
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    if os.path.exists(dict_path):
        word_to_id = _load_vocab(dict_path)
    else:
        word_to_id = _build_vocab(train_path)
        _save_vocab(word_to_id,dict_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    # 从PTB数据迭代的读取batch大小的数据
    # 返回包含这些数据的Tensor

    # Args:
    #   raw_data    - ptb_raw_data 的返回值
    #               - list of integer
    #   batch_size  - batch 大小
    #   num_steps   - LSTM 时间序列长度
    #   name: the name of this operation (optional).

    # Returns:
    #   两组[batch_size, num_steps]大小的Tensor
    #   第二个元素为左边相同数据时移+1

    # Raises:
    # tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.

    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(
            raw_data, name="raw_data", dtype=tf.int32)
        #总数据长度
        data_len = tf.size(raw_data)
        #batch总数
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y

if __name__ == "__main__":
    train_data, valid_data, test_data, vocabulary = ptb_raw_data("../data")
    print(1)
