import pickle
import tensorflow as tf
import numpy as np



fr = open('ext_jar/data/allword2vec.pkl','rb')    #open的参数是pkl文件的路径
# Label = open('ext_jar/data/label.pkl','rb')
#wordindex = open('ext_jar/data/AllWordIndex.pkl','rb')

inf = pickle.load(fr) #读取pkl文件的内容
# label1 = pickle.load(Label)
#WordIndex = pickle.load(wordindex)

# WordIndex =list(WordIndex)
# WordIndexshape = len(WordIndex)
# qwe = WordIndex[0]
# # fr.close()
# Label.close()

#print(inf)
print(inf.shape)
# WordIndex.close
#
# BATCH_SIZE = 50
# LR = 0.001

#定义batch
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.批量数据batchsize生成器
    定义一个函数，输出batch样本，参数为data（包括feature和label），batchsize，epoch
    """
    data = np.array(data)#全部数据转化为array
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1#每个epoch有多少个batch，个数
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]# shuffled_data按照上述乱序得到新的样本
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):#开始生成batch
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)#这里主要是最后一个batch可能不足batchsize的处理
            yield shuffled_data[start_index:end_index]
            #yield，在for循环执行时，每次返回一个batch的data，占用的内存为常数


##TEXTCNN训练框架
class TextCNN(object):#定义了1个TEXTCNN的类，包含一张大的graph
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    embedding层，卷积层，池化层，softmax层
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):#定义各种输入参数，这里的输入是句子各词的索引？

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        #定义一个operation，名称input_x,利用参数sequence_length，None表示样本数不定，
        #不一定是一个batchsize，训练的时候是，验证的时候None不是batchsize
        #这是一个placeholder，
        #数据类型int32，（样本数*句子长度）的tensor，每个元素为一个单词
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        #这个placeholder的数据输入类型为float，（样本数*类别）的tensor
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        #placeholder表示图的一个操作或者节点，用来喂数据，进行name命名方便可视化

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        #l2正则的初始化，有点像sum=0
        #其实softmax是需要的

        # Embedding layer  自己自动生成训练词向量
        
        #参见
        with tf.device('/cpu:0'), tf.name_scope("embedding"):#封装了一个叫做“embedding'的模块，使用设备cpu，模块里3个operation
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")#operation1，一个（词典长度*embedsize）tensor，作为W，也就是最后的词向量
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            #operation2，input_x的tensor维度为[none，seq_len],那么这个操作的输出为none*seq_len*em_size
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            #增加一个维度，变成，batch_size*seq_len*em_size*channel(=1)的4维tensor，符合图像的习惯

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []#空list
        for i, filter_size in enumerate(filter_sizes):#比如（0，3），（1，4），（2，5）
            with tf.name_scope("conv-maxpool-%s" % filter_size):#循环第一次，建立一个名称为如”conv-ma-3“的模块
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                #operation1，没名称，卷积核参数，高*宽*通道*卷积个数
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                #operation2，名称”W“，变量维度filter_shape的tensor
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                #operation3，名称"b",变量维度卷积核个数的tensor
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],#样本，height，width，channel移动距离
                    padding="VALID",
                    name="conv")
                #operation4，卷积操作，名称”conv“，与w系数相乘得到一个矩阵
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #operation5，加上偏置，进行relu，名称"relu"
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],   #经过卷积后的池化层核
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                #每个卷积核和pool处理一个样本后得到一个值，这里维度如batchsize*1*1*卷积核个数
                #三种卷积核，appen3次

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        #operation，每种卷积核个数与卷积核种类的积
        self.h_pool = tf.concat(pooled_outputs, 3)
        #operation，将outpus在第4个维度上拼接，如本来是128*1*1*64的结果3个，拼接后为128*1*1*192的tensor
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        #operation，结果reshape为128*192的tensor

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        #添加一个"dropout"的模块，里面一个操作，输出为dropout过后的128*192的tensor

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):#添加一个”output“的模块，多个operation
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            #operation1，系数tensor，如192*2，192个features分2类，名称为"W"，注意这里用的是get_variables
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            #operation2,偏置tensor，如2，名称"b"
            l2_loss += tf.nn.l2_loss(W)
            #operation3，loss上加入w的l2正则
            l2_loss += tf.nn.l2_loss(b)
            #operation4，loss上加入b的l2正则
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            #operation5，scores计算全连接后的输出，如[0.2,0.7]名称”scores“
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            #operations，计算预测值，输出最大值的索引，0或者1，名称”predictions“

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):#定义一个”loss“的模块
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            #operation1，定义losses，交叉熵，如果是一个batch，那么是一个长度为batchsize1的tensor？
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            #operation2，计算一个batch的平均交叉熵，加上全连接层参数的正则

        # Accuracy
        with tf.name_scope("accuracy"):#定义一个名称”accuracy“的模块
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            #operation1，根据input_y和predictions是否相同，得到一个矩阵batchsize大小的tensor
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            #operation2，计算均值即为准确率，名称”accuracy“

