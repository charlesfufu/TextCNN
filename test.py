from jpype import *
import platform
import re
import tensorflow as tf
import pickle
import pymysql.cursors
import codecs
from ext_jar.jar_service import*
from gensim.models import word2vec
import numpy as np
from sklearn.model_selection import train_test_split
'''
随机初始化embedding词向量，让TEXTCNN去自己拟合
'''

def ConnectSQL(host1,port1,user1,passwd1,db1,charset1):
    # 连接数据库
    connect = pymysql.Connect(
        host=host1,
        port=port1,
        user=user1,
        passwd=passwd1,
        db=db1,
        charset=charset1
    )

    # 获取游标
    cursor = connect.cursor()
    sql = "SELECT ClassLabel,Content FROM `Corpus_TextClassifier_Estate` "
    cursor.execute(sql)
    connect.commit()
    num = 0
    with codecs.open('ext_jar/data/segmentsql.txt', mode='w', encoding="UTF-8") as f:
       for row in cursor.fetchall():
           #print(row)
           Segrow = segment(row[1])
           result = Segrow["segmented_result"]
           result1 = row[0] + ' ' + result
           #print(result1)
           f.write(result1)
           f.write('\n')
           num += 1
           print('跑了的文章数为：',num)

    cursor.close()
    connect.close()


#定义batch
#x_y_data is the list of x y
def x_y_data_iter(x_y_data, batch_size = 100, num_epochs = 10, shuffle = True):
    x, y = x_y_data
    x_data = np.array(x)
    y_data = np.array(y)

    data_size = len(x_data)
    num_batches_per_epoch = int((len(x_data)-1)/batch_size) + 1#每个epoch有多少个batch，个数
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data_x = x_data[shuffle_indices]# shuffled_data按照上述乱序得到新的样本
            shuffled_data_y = y_data[shuffle_indices]
        else:
            shuffled_data_x = x_data
            shuffled_data_y = y_data
        for batch_num in range(num_batches_per_epoch):#开始生成batch
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)#这里主要是最后一个batch可能不足batchsize的处理
            yield shuffled_data_x[start_index:end_index], shuffled_data_y[start_index:end_index]


##TEXTCNN训练框架
class TextCNN(object):  # 定义了1个TEXTCNN的类，包含一张大的graph
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    embedding层，卷积层，池化层，softmax层
    """

    def __init__(
            self, sequence_length=12859, num_classes=6, vocab_size=37283,
            embedding_size=100, filter_sizes=[3,4,5], num_filters=3, l2_reg_lambda=0.0):  # 定义各种输入参数，这里的输入是句子各词的索引？
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")

        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        # 这个placeholder的数据输入类型为float，（样本数*类别）的tensor
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # placeholder表示图的一个操作或者节点，用来喂数据，进行name命名方便可视化

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # l2正则的初始化，有点像sum=0
        # 其实softmax是需要的

        # Embedding layer  自己自动生成训练词向量

        # 参见
        with tf.device('/cpu:0'), tf.name_scope("embedding"):  # 封装了一个叫做“embedding'的模块，使用设备cpu，模块里3个operation
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")  # operation1，一个（词典长度*embedsize）tensor，作为W，也就是最后的词向量
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # operation2，input_x的tensor维度为[none，seq_len],那么这个操作的输出为none*seq_len*em_size
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # 增加一个维度，变成，batch_size*seq_len*em_size*channel(=1)的4维tensor，符合图像的习惯

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []  # 空list
        for i, filter_size in enumerate(filter_sizes):  # 比如（0，3），（1，4），（2，5）
            with tf.name_scope("conv-maxpool-%s" % filter_size):  # 循环第一次，建立一个名称为如”conv-ma-3“的模块
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # operation1，没名称，卷积核参数，高*宽*通道*卷积个数
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # operation2，名称”W“，变量维度filter_shape的tensor
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # operation3，名称"b",变量维度卷积核个数的tensor
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],  # 样本，height，width，channel移动距离
                    padding="VALID",
                    name="conv")
                # operation4，卷积操作，名称”conv“，与w系数相乘得到一个矩阵
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # operation5，加上偏置，进行relu，名称"relu"
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],  # 经过卷积后的池化层核
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                # 每个卷积核和pool处理一个样本后得到一个值，这里维度如batchsize*1*1*卷积核个数
                # 三种卷积核，appen3次

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        # operation，每种卷积核个数与卷积核种类的积
        self.h_pool = tf.concat(pooled_outputs, 3)
        # operation，将outpus在第4个维度上拼接，如本来是128*1*1*64的结果3个，拼接后为128*1*1*192的tensor
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # operation，结果reshape为128*192的tensor

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        # 添加一个"dropout"的模块，里面一个操作，输出为dropout过后的128*192的tensor

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):  # 添加一个”output“的模块，多个operation
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            # operation1，系数tensor，如192*2，192个features分2类，名称为"W"，注意这里用的是get_variables
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # operation2,偏置tensor，如2，名称"b"
            l2_loss += tf.nn.l2_loss(W)
            # operation3，loss上加入w的l2正则
            l2_loss += tf.nn.l2_loss(b)
            # operation4，loss上加入b的l2正则
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # operation5，scores计算全连接后的输出，如[0.2,0.7]名称”scores“
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # operations，计算预测值，输出最大值的索引，0或者1，名称”predictions“

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):  # 定义一个”loss“的模块
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            # operation1，定义losses，交叉熵，如果是一个batch，那么是一个长度为batchsize1的tensor？
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            # operation2，计算一个batch的平均交叉熵，加上全连接层参数的正则

        # Accuracy
        with tf.name_scope("accuracy"):  # 定义一个名称”accuracy“的模块
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # operation1，根据input_y和predictions是否相同，得到一个矩阵batchsize大小的tensor
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # operation2，计算均值即为准确率，名称”accuracy“

        # opt
        self.trsin_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)



    @staticmethod
    def train():
        X_train, X_test, Y_train, Y_test = train_test_split(
            x_data, y_data, test_size= 1 - train_sample_ratio, random_state=0)

        textcnn_ext = TextCNN()
        train_gen = x_y_data_iter([X_train, Y_train], num_epochs=10000, batch_size=10)
        test_gen = x_y_data_iter([X_test, Y_test], num_epochs=1, batch_size=50)

        times = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            while True:
                try:
                    x_train, y_train = train_gen.__next__()
                except:
                    print("train_epoch end return")
                    return

                _, loss, acc = sess.run([textcnn_ext.trsin_op, textcnn_ext.loss, textcnn_ext.accuracy],
                         feed_dict={
                             textcnn_ext.input_y: y_train,
                             textcnn_ext.input_x: x_train,
                             textcnn_ext.dropout_keep_prob: 0.7
                         })
                times += 1
                if times % 100 == 0:
                    print("train loss:{} acc:{}".format(loss, acc))

                if times % 1000 == 0:
                    try:
                        x_test, y_test = test_gen.__next__()
                    except:
                        print("test_epoch end")
                        test_gen = x_y_data_iter([X_test, Y_test], num_epochs=1, batch_size=50)

                    loss, acc = sess.run([textcnn_ext.loss, textcnn_ext.accuracy],
                                         feed_dict={
                                             textcnn_ext.input_y: y_test,
                                             textcnn_ext.input_x: x_test,
                                             textcnn_ext.dropout_keep_prob: 1.0
                                         })
                    print("test loss:{} acc:{}".format(loss, acc))






if __name__ == "__main__":

   #列出标签和数据
   label = []
   value = []
   labelnum = []
   #ConnectSQL(host1='10.20.34.113',port1=3306,user1='root',passwd1='r#dcenter9',db1='zhaoqc24049',charset1='utf8')
   with codecs.open('ext_jar/data/segmentsql.txt', mode='r', encoding="UTF-8") as h:
        for line in h:
            #list = line.split()
            label.append(line.split()[0])
            value.append(line.split()[1:])
   finlabel = set(label)
   finlabel = list(finlabel)
   for i in range(len(label)):
       labelnum.append(finlabel.index(label[i]))

   # feature_word2vec = codecs.open(filename="ext_jar/data/allword2vec.pkl", mode='wb')
   # label = codecs.open(filename="ext_jar/data/label.pkl", mode='wb')
   # pickle.dump(np.array(labelnum), label)


   model_2 = word2vec.Word2Vec.load("word2vec.model")
   Vocabulary = list(model_2.wv.vocab.keys())
   '''
   VocIndex = []
   ALLVocIndex = []
   for l in range(len(value)):
       for i in range(len(value[l])):
           if value[l][i] in Vocabulary:
               VocIndex.append(Vocabulary.index(value[l][i]))
       ALLVocIndex.append(VocIndex)
       VocIndex = []
   '''
   vocabulary = codecs.open(filename="ext_jar/data/ALLVocIndex.pkl", mode='rb')
   ALLVocIndex = pickle.load(vocabulary)
   length = []
   for i in range(len(ALLVocIndex)):
       length.append(len(ALLVocIndex[i]))
   maxlength = max(length)

   xdata = np.zeros((len(value),maxlength),dtype=int)
   for l in range(len(ALLVocIndex)):
       for i in range(len(ALLVocIndex[l])):
           xdata[l][i] = int(ALLVocIndex[l][i])
   labelone_hot = np.zeros((len(value), 6))

   # 训练和预测的0-1标签
   with open('ext_jar/data/label.pkl', 'rb') as f:
       y_data = pickle.load(f)
   for i in range(len(value)):
       labelone_hot[i][y_data[i]] = 1

   x_data = xdata
   y_data = labelone_hot
   train_sample_ratio = 0.9
   TextCNN.train()






















'''
   for l in range(3):
      for i in range(len(value[l])):
           if value[l][i] in Voca:
               WordIndex.append(Voca.index(value[l][i]))
      AllWordIndex.append(WordIndex)



   print(AllWordIndex[0],len(AllWordIndex[0]))
   print(len(value[0]))




   AllWordIndex = AllWordIndex
   AllWordIndex1 = codecs.open(filename="ext_jar/data/AllWordIndex.pkl", mode='wb')
   pickle.dump(AllWordIndex, AllWordIndex1)

   sentences = word2vec.Text8Corpus(u"ext_jar/data/segmentsql.txt")
   model = word2vec.Word2Vec(sentences,sg=1,size=100,window=5)
   model.save(u"word2vec.model")
   得到所有词向量的字典
   Allvocabulary = list(model_2.wv.vocab.keys())
   vocabulary = codecs.open(filename="ext_jar/data/Allvocabulary.pkl", mode='wb')
   pickle.dump(Allvocabulary, vocabulary)



       word2veclist1.append(model[value[0][i]])
   word2vecvoca =list(model.wv.vocab.keys())


   print(word2vecvoca)
   print(model.wv.vocab.keys())
   print(len(model.wv.vocab.keys()))
   print(model)
   #print(model.vocabulary())
   print(value[0])

   print(a)


'''


