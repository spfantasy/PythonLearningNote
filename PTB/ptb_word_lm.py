# -*- coding: UTF-8 -*-
"""
#务必修改 vocab_size 大于等于 词典大小+1
#务必修改 num_steps 小于等于 同一句长
#务必修改 batch_size 为偶数
The hyperparameters used in the model:
- init_scale    - 权重初始化的方差
- learning_rate - 初始学习速率
- max_grad_norm - 梯度上限
- num_layers    - LSTM 层数
- num_steps     - LSTM 时间维展开长度
- hidden_size   - LSTM 
- max_epoch     - 当前学习速率学习的最大epoch数
- max_max_epoch - 总epoch数
- keep_prob     - 1-dropout几率
- lr_decay      - 学习速率衰减
- batch_size    - batch大小

#官方测试数据
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
官方运行命令
$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# 查看类初始化参数
import inspect
import time
import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging
# 命令行参数
flags.DEFINE_string(
    "model", "small",  # 选择运行的参数
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "../data/",  # simple-examples/data/",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", "../model/",
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBInput(object):
    # 输入

    def __init__(self, config, data, label, name=None):
        # 强制batch_size 为偶数
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.labels = reader.ptb_producer(
            data, label, batch_size, num_steps, name=name)

class PTBModel(object):
    # 包裹Tensors的类
    #                   是否训练中   数据信息  数据流

    def __init__(self, is_training, config, input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        # forget_bias = 1.0结果稍好但需要调参
        def lstm_cell():
            # 建立[1,hidden_size]的LSTM序列
            if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(
                    size, forget_bias=0.0, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                    size, forget_bias=0.0, state_is_tuple=True)
        attn_cell = lstm_cell
        # 需要dropout时用DropoutWrapper包裹序列
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)
        # cell为多个LSTM层堆叠的结构
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        # 初始化batch_size个cell的state
        self._initial_state = cell.zero_state(batch_size, data_type())
        # 查询每个单词对应的向量
        #[batch_size, num_steps]->[batch_size, num_steps, vocab_dim]
        with tf.device("/cpu:0"):
            self._embedding = embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(
        #     cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # 所有样本的第time_step个词输入到各自的cell里
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                # outputs    - list of tensor[20,2]
                outputs.append(cell_output)
        # output  - Tensor[1,hidden_size(size)]
        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1,size])
        fc1_w = tf.get_variable(
            "fc1_w", [size, 1], dtype=data_type())
        fc1_b = tf.get_variable(
            "fc1_b", [1], dtype=data_type())
        fc1 = tf.sigmoid(tf.matmul(output, fc1_w) + fc1_b)
        fc1 = tf.reshape(fc1, [batch_size, num_steps* 1])
        fc2_w = tf.get_variable(
            "fc2_w", [num_steps* 1, 1], dtype=data_type())
        fc2_b = tf.get_variable(
            "fc2_b", [1], dtype=data_type()) 
        _labels = tf.matmul(fc1, fc2_w) + fc2_b       
        self._labels = _labels = tf.sigmoid(_labels)
        # 使用input_.labels 和 logits计算 loss
        loss = tf.square(_labels - tf.cast(input_.labels,data_type()))
        #设定_labels的阈值
        _labels_thres = tf.constant(np.full([_labels.shape[0].value,_labels.shape[1].value], .5),dtype=data_type())
        # _labels二值化
        prediction = tf.cast(tf.less(_labels_thres,_labels), tf.int32)
        #计算准确度
        correct_prediction = tf.equal(prediction,input_.labels)
        self._accuracy = accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # update the cost variables
        self._cost = cost = tf.reduce_sum(loss)
        self._final_state = state

        if not is_training:
            return
        # Tensor of Learning Rate
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def embedding(self):
        return self._embedding

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def labels(self):
        return self._labels

    @property
    def cost(self):
        return self._cost

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):
    """Small config."""
    init_scale = 1e-3#0.1
    learning_rate = 0.1#1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 80
    hidden_size = 256
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 5000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 80
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 5000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 80
    hidden_size = 256#1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 5000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 5000


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    accuracys = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "accuracy": model.accuracy,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        accuracy = vals["accuracy"]

        costs += cost
        iters += model.input.num_steps
        accuracys += accuracy

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f error: %.3f accu: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, costs / step,
                    accuracys/step,
                   iters * model.input.batch_size / (time.time() - start_time)))

    	print(session.run(model.embedding).reshape([-1,1])[:10])##
    return accuracys/model.input.epoch_size


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

# 张量流运行


def main(_):
    # 检查CMD或者文件头默认路径是否正确
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, train_label_data, valid_label_data, test_label_data, _ = raw_data
    # 导入hyperparameter
    config = get_config()
    eval_config = get_config()
    # eval_config.batch_size = 1

    with tf.Graph().as_default():
        # 初始化LSTM参数
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(
                config=config, data=train_data, label = train_label_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config,
                             input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(
                config=config, data=valid_data, label = valid_label_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False,
                                  config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config,
                                  data=test_data, label = test_label_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config,
                                 input_=test_input)
        # 自动读取已有的checkpoint
        # 自带saver和summary模块，不需重复创建
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i +
                                                  1 - config.max_epoch, 0.0)
                # 对LearningRate进行下滑
                m.assign_lr(session, config.learning_rate * lr_decay)
                # 读取当前Learning Rate
                print("Epoch: %d Learning rate: %.3f" %
                      (i + 1, session.run(m.lr)))
                # 进行一个epoch的训练
                train_accuracy = run_epoch(session, m, eval_op=m.train_op,
                                        verbose=True)
                print("Epoch: %d Train accuracy: %.3f" %
                      (i + 1, train_accuracy))
                # 进行一个epoch的混乱度验证
                valid_accuracy = run_epoch(session, mvalid)
                print("Epoch: %d Valid accuracy: %.3f" %
                      (i + 1, valid_accuracy))
            # 测试集
            test_accuracy = run_epoch(session, mtest)
            print("Test accuracy: %.3f" % test_accuracy)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path,
                              global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
