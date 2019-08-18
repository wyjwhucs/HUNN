# -*- coding: UTF-8 -*-
import os
import time
import math
import re
import yaml
import datetime
import jpype
import copy
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

def get_datasets_MR(positive_data_file, negative_data_file):
    positive_examples = list(open(positive_data_file, "r",encoding = 'utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r",encoding = 'utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    datasets = dict()
    datasets['data'] = positive_examples + negative_examples
    target = [0 for x in positive_examples] + [1 for x in negative_examples]
    datasets['target'] = target
    datasets['target_names'] = ['positive_examples', 'negative_examples']
    return datasets

def get_datasets_multilabel(train_file, test_file):   
    # 载入多标签文本数据集，将数据拆分为单词和标签，返回文本和标签。
    text=[]
    label=[]
    #读取文件的每一行
    train_examples = list(open(train_file, "r",encoding = 'utf-8').readlines())
    train_examples = [s.strip() for s in train_examples]#去掉空格
    for s in train_examples:
        ss= re.split(":",s)
        label.append(ss[0])#标签
        text.append(ss[1])#文本

    #读取文件的每一行
    test_examples = list(open(test_file, "r",encoding = 'utf-8').readlines())
    test_examples = [s.strip() for s in test_examples]#去掉空格
    for s in test_examples:
        ss= re.split(":",s)
        label.append(ss[0])#标签
        text.append(ss[1])#文本

    datasets = dict()
    datasets['data'] = text
    datasets['target'] = label
    return datasets
 
def get_datasets_multilabel_dev(train_file, test_file, dev_file):   
    # 载入多标签文本数据集，将数据拆分为单词和标签，返回文本和标签。
    text=[]
    label=[]
    #读取文件的每一行
    train_examples = list(open(train_file, "r",encoding = 'utf-8').readlines())
    train_examples = [s.strip() for s in train_examples]#去掉空格
    for s in train_examples:
        ss= re.split(":",s)
        label.append(ss[0])#标签
        text.append(ss[1])#文本

    #读取文件的每一行
    test_examples = list(open(test_file, "r",encoding = 'utf-8').readlines())
    test_examples = [s.strip() for s in test_examples]#去掉空格
    for s in test_examples:
        ss= re.split(":",s)
        label.append(ss[0])#标签
        text.append(ss[1])#文本

    #读取文件的每一行
    dev_examples = list(open(dev_file, "r",encoding = 'utf-8').readlines())
    dev_examples = [s.strip() for s in dev_examples]#去掉空格
    for s in dev_examples:
        ss= re.split(":",s)
        label.append(ss[0])#标签
        text.append(ss[1])#文本
        
        
    datasets = dict()
    datasets['data'] = text
    datasets['target'] = label
    return datasets


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    # num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    num_batches_per_epoch = data_size // batch_size
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            # if end_index - start_index != batch_size:
                # yield shuffled_data[end_index-batch_size:end_index]
            yield shuffled_data[start_index:end_index]

def load_data_label(datasets):#回归任务
    """
    Load data and label
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    # x_text = [clean_str(sent) for sent in x_text]
    x_text = [sent for sent in x_text]
    # Generate regressor label
    label = []
    for i in range(len(x_text)):
        score = datasets['target'][i]
        label.append([score])
    y = np.array(label)
    return [x_text, y]

def load_data_labels_MR(datasets):#分类任务，2个标签数据
    #原始标签数据：正、负
    #输出标签y为一个2维向量
    x_text = datasets['data']
    x_text = [clean_str(sent) for sent in x_text]
    # x_text = [sent for sent in x_text]
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]

def load_data(datasets):
    """
    Load data without labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    # x_text = [clean_str(sent) for sent in x_text]
    x_text = [sent for sent in x_text]
    return x_text


def LOAD_DATA():
    #读取预定义数据
    with open("config.yml",'r', encoding='UTF-8') as ymlfile:
        cfg = yaml.load(ymlfile)
    #数据集名称：dataset_name
    dataset_name = cfg["datasets"]["default"]#default:选择数据集
    #词向量地址：embedding_path
    embedding_path=cfg['word_embeddings']
    #载入数据...
    datasets = None
    if dataset_name == "MR":
        datasets = get_datasets_MR(
                cfg["datasets"][dataset_name]["positive_data_file"]["path"],                              
                cfg["datasets"][dataset_name]["negative_data_file"]["path"])
    if dataset_name=='MR':
        x_text, y = load_data_labels_MR(datasets)#载入标签
    return [x_text, y, embedding_path,dataset_name]
    
def Vocab_proces(x_text):
    #这一步将原始文本(x_text)转换为矩阵(x)表示，x:10662*56
    max_document_length = max([len(x.split(" ")) for x in x_text])+1#文档的长度44
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    #如果文本的长度大于最大长度，那么它会被剪切，反之则用0填充。 
    print("原始数据最大文本长度: {:d}".format(max_document_length))
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    return [x,vocab_processor]

def DATA_split(x,y,test_percentage):
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))#生成10662个顺序打乱的数据
    xs = x[shuffle_indices]#打乱后的文本矩阵
    ys = y[shuffle_indices]#打乱后的标签矩阵
    # 将数据分为训练集和测试集,根据设置确定的测试比例，获取测试样本数量
    R = -1 * int(test_percentage * float(len(y)))#测试样本数
    #这一步得到了训练文本矩阵(x_train)和训练文本标签(x_dev)，以及测试文本矩阵(y_train)和文本标签(y_dev)
    x_train, x_dev = xs[:R], xs[R:]#词向量矩阵，9596个训练样本和1066个测试集
    y_train, y_dev = ys[:R], ys[R:]#标签，9596个训练样本和1066个测试集
    return [x_train, x_dev,y_train, y_dev]

def HUI_DEL_Zero(x_train, y_train):
    x_0=[]#存储数据全为0的索引
    for i in range(len(x_train)):
        if x_train[i][0]==0:
            x_0.append(i)#将索引添加到x_0中
    for i in reversed(x_0):#将list倒序遍历
        x_train = np.delete(x_train, i, 0)  # 删除x_train的第i行
        y_train = np.delete(y_train, i, 0)  # 删除y_train的第i行
    return [x_train, y_train]

def collect(lst):
    return dict(collections.Counter(lst))

def Text_To_HUI(x_train, y_train):
    trans=""#存储事务数据
    for i in range(len(x_train)):
        temp=x_train[i,:]
        trans_tp=""
        for key in temp:
            trans_tp=trans_tp+str(key)+' '
        trans=trans+"&"+trans_tp
    trans=trans[1:]#去掉第一个&符号
    
    quantity=""#存储项与数量(效用值)
    for i in range(len(x_train)):
        temp=x_train[i,:]
        d2 = collect(temp)#统计出现次数据
        quantity_tp=""
        for key,value in d2.items():
            if key==0:
                pass
            else:
                quantity_tp=quantity_tp+str(key)+' '+str(value)+' '
        quantity=quantity+"&"+quantity_tp
    quantity=quantity[1:]#去掉第一个&符号
    
    label=""#存储对应对的标签
    for i in range(len(x_train)):
        temp=y_train[i,:]
        label_tp=""
        for key in temp:
            label_tp=label_tp+str(key)+' '
        label=label+"&"+label_tp
    label=label[1:]#去掉第一个&符号
    return [trans, quantity, label]
    
def HUI_To_Text(result,x_dev,y_dev,Test_Filter):
    result_split=result.split(':')
    str_trans=result_split[0].split('&')
    str_label=result_split[1].split('&')
    str_trans.pop()#删除最后一个元素,原始数据x
    str_label.pop()#删除最后一个元素,标签数据y
    x_temp=[]
    col=0
    for tran in str_trans:
        tran_temp=re.split(r" +",tran[:-1])#去掉最后一个空格，再按照多个空格分割
        tran_temp = [ int(i) for i in tran_temp ]#tran_temp:将str类型转的是为int型
        if len(tran_temp)>col:
            col=len(tran_temp)
        x_temp.append(tran_temp)
    row=len(x_temp)
    
    xx=np.zeros((row, col), dtype=np.int)
    for i in range(len(x_temp)):
        for j in range(len(x_temp[i])):
            xx[i][j]= x_temp[i][j]
    y_temp=[]
    col_y=0
    for lab in str_label:
        lab_temp=re.split(r" +",lab[:-1])#去掉最后一个空格，再按照多个空格分割
        lab_temp = [ int(i) for i in lab_temp ]#tran_temp:将str类型转的是为int型
        if len(lab_temp)>col_y:
            col_y=len(lab_temp)
        y_temp.append(lab_temp)
    row_y=len(y_temp)
    yy=np.zeros((row_y, col_y), dtype=np.int)
    for i in range(len(y_temp)):
        for j in range(len(y_temp[i])):
            yy[i][j]= y_temp[i][j]
    #del x_train,y_train#删除原始文件数据
    x_train=xx#保留经过HUI过虑后的文本数据
    y_train=yy#保留经过HUI过虑后的标签数据  
    if  Test_Filter:#如果使用高效用词集过虑测试数据
        tran_filter_set= set()#获取训练集中所有单词的编号
        for i in range(0,len(x_train)): 
            tran_filter_set=tran_filter_set|set(x_train[i].tolist())
        lise_x_dev=[]
        lise_x_dev_col=0
        for j in range(0,len(x_dev)): #1066个测试数据
            x_dev_j=x_dev[j]#一行测试事务数据
            lise_x_dev_row=[]
            for k in range(0,len(x_dev_j)):
                if x_dev_j[k]==0:
                    break;
                else:
                   if x_dev_j[k] in tran_filter_set:#该单词在训练集中存在
                       lise_x_dev_row.append(x_dev_j[k])
            lise_x_dev.append(lise_x_dev_row)
            if len(lise_x_dev_row)>lise_x_dev_col:
                lise_x_dev_col=len(lise_x_dev_row)#取最大的宽
        lise_x_dev_row=len(lise_x_dev)
        xx_dev=np.zeros((lise_x_dev_row, lise_x_dev_col), dtype=np.int)
        for i in range(len(lise_x_dev)):
            for j in range(len(lise_x_dev[i])):
                xx_dev[i][j]= lise_x_dev[i][j]      
                
        xx_dev_0=[]#存储数据全为0的索引
        for i in range(len(xx_dev)):
            if xx_dev[i][0]==0:
                xx_dev_0.append(i)#将索引添加到xx_dev_0中
        for i in reversed(xx_dev_0):#将list倒序遍历，相当于从后向前删
            xx_dev = np.delete(xx_dev, i, 0)  # 删除xx_dev的第i行
            y_dev = np.delete(y_dev, i, 0)  # 删除y_dev的第i行
            print("删除xx_dev,y_dev的全0元素后的数据: {:d}".format(len(xx_dev)))

        del x_dev#删除原始测试数据
        x_dev=xx_dev#保留经过HUI过虑后的测试数据
    if len(x_dev[0])>len(x_train[0]):#训练数据需要补0
        print("测试数据文本长度长,训练数据需要补0: 测试{:d},训练{:d}".format(len(x_dev[0]),len(x_train[0])))
        x_sup_col=len(x_dev[0])-len(x_train[0])
        x_sup_row=len(x_train)
        x_train_sup=np.zeros((x_sup_row, x_sup_col),dtype=np.int)
        x_train=np.hstack((x_train, x_train_sup))
    elif len(x_dev[0])==len(x_train[0]):
        print("训练数据和测试数据的文本长度相等: 测试{:d},训练{:d}".format(len(x_dev[0]),len(x_train[0])))
    elif len(x_dev[0])<len(x_train[0]):#测试数据需要补0
        print("训练数据文本长度长,测试数据需要补0: 测试{:d},训练{:d}".format(len(x_dev[0]),len(x_train[0])))
        x_sup_col=len(x_train[0])-len(x_dev[0])
        x_sup_row=len(x_dev)
        x_dev_sup=np.zeros((x_sup_row, x_sup_col),dtype=np.int)
        x_dev=np.hstack((x_dev, x_dev_sup))
    return [x_train, y_train, x_dev, y_dev]

def ToSentence(x_view,vocab_processor):#将ID转换为单词
    x_view_Sentence=[]#存储数据全为0的索引
    for i in x_view:
        x_view_Sent=[]#存储数据全为0的索引
        for j in i:
            if j == 0:#如果没有单词跳出当前句子
                continue
            else:
                voc=vocab_processor.vocabulary_.reverse(j)#根据ID查找单词
                x_view_Sent.append(voc)
        x_view_Sentence.append(x_view_Sent)
    return x_view_Sentence

class TextHUNN(object):
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, batch_size, l2_reg_lambda=0.5): 
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        l2_loss = tf.constant(0.0)
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # When trainable parameter equals True the embedding vector is non-static, otherwise is static
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W", trainable=True)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)  # [None, sequence_length, embedding_size]
        with tf.name_scope('rcnn'):
            # define weights here
            self.initializer = tf.random_normal_initializer(stddev=0.1)
            self.left_side_first_word = tf.get_variable("left_side_first_word", shape=[batch_size, embedding_size], initializer=self.initializer)
            self.right_side_last_word = tf.get_variable("right_side_last_word", shape=[batch_size, embedding_size], initializer=self.initializer)
            self.W_l = tf.get_variable("W_l", shape=[embedding_size, embedding_size], initializer=self.initializer)
            self.W_r = tf.get_variable("W_r", shape=[embedding_size, embedding_size], initializer=self.initializer)
            self.W_sl = tf.get_variable("W_sl", shape=[embedding_size, embedding_size], initializer=self.initializer)
            self.W_sr = tf.get_variable("W_sr", shape=[embedding_size, embedding_size], initializer=self.initializer)
            def get_context_left(context_left, embedding_previous):
                left_c = tf.matmul(context_left, self.W_l)  #context_left:[batch_size,embed_size]; W_l:[embed_size,embed_size]
                left_e = tf.matmul(embedding_previous, self.W_sl)  #embedding_previous; [batch_size,embed_size]
                left_h = left_c + left_e
                context_left = tf.nn.relu(left_h, name="relu") # [None,embed_size]
                return context_left
            def get_context_right(context_right, embedding_afterward):
                right_c = tf.matmul(context_right, self.W_r)
                right_e = tf.matmul(embedding_afterward, self.W_sr)
                right_h = right_c + right_e
                context_right = tf.nn.relu(right_h, name="relu")
                return context_right
            embedded_words_split = tf.split(self.embedded_chars, sequence_length, axis=1) #sentence_length * [None,1,embed_size]
            embedded_words_squeezed = [tf.squeeze(x, axis=1) for x in embedded_words_split] #sentence_length * [None,embed_size]
            embedding_previous = self.left_side_first_word
            context_left_previous = tf.zeros((batch_size, embedding_size))
            context_left_list=[]
            for i, current_embedding_word in enumerate(embedded_words_squeezed): #sentence_length * [None,embed_size]
                context_left = get_context_left(context_left_previous, embedding_previous) #[None,embed_size]
                context_left_list.append(context_left) #append result to list
                embedding_previous = current_embedding_word #assign embedding_previous
                context_left_previous = context_left #assign context_left_previous
            embedded_words_squeezed2 = copy.copy(embedded_words_squeezed)
            embedded_words_squeezed2.reverse()
            embedding_afterward = self.right_side_last_word
            context_right_afterward = tf.zeros((batch_size, embedding_size))
            context_right_list=[]
            for j, current_embedding_word in enumerate(embedded_words_squeezed2):
                context_right = get_context_right(context_right_afterward, embedding_afterward)
                context_right_list.append(context_right)
                embedding_afterward = current_embedding_word
                context_right_afterward = context_right
            output_list=[]
            for index, current_embedding_word in enumerate(embedded_words_squeezed):
                representation = tf.concat([context_left_list[index], current_embedding_word, context_right_list[index]], axis=1)
                output_list.append(representation) #shape:sentence_length * [None,embed_size*3]
            outputs = tf.stack(output_list, axis=1) #shape:[None,sentence_length,embed_size*3]
            self.output = tf.reduce_max(outputs, axis=1) #shape:[None,embed_size*3]
        with tf.name_scope("dropout"):
            self.rnn_drop = tf.nn.dropout(self.output, self.dropout_keep_prob)
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[embedding_size*3, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.rnn_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

test_percentage= 0.1# "测试数据比率"
stop_words=False#"是否使用停用词过虑 True/False"
Filter_Mode= 3 # 高效用词集过虑模式  默认:1/2/3"
Test_Filter= True # 是否使用高效用词集过虑测试集:True/False"
hidden_size = 128 # 隐藏层神经元数  默认: 128"
hidden_layers = 2 # 隐藏层数  默认: 2"
filter_sizes = "3,4,5"#滤波器大小  默认: '3,4,5'"
num_filters = 128 # 每种滤波器的数量  默认: 128"
rnn_size = 300 # 运行层神经元数  默认: 300"
num_rnn_layers = 3 # 运行层数  默认: 3"
dropout_keep_prob = 0.5 # 退出保持率  默认: 0.5"
l2_reg_lambda = 1.0 # L2正则化lambda值  默认: 0.0"
batch_size = 64 # 批量大小  默认: 64"
num_epochs = 20 # 训练迭代次数  默认: 20"
evaluate_every = 100 # 100次后产生评估模型  默认: 100"
checkpoint_every = 100 # 100次后保存模型  默认: 100"
num_checkpoints = 5 # 要存储的检查点数量  默认: 5"
grad_clip = 5 # 防止梯度爆炸"
decay_coefficient = 2.5 # 衰变系数  默认: 2.5"
allow_soft_placement = True # 允许设备放置"
log_device_placement = False # 操作在设备上的日志放置"

x_text, y, embedding_path, dataset_name = LOAD_DATA()

jarpath = 'D:\DataMining\TC\JAR'
jpype.startJVM(jpype.getDefaultJVMPath(), "-Djava.ext.dirs=%s" % jarpath)
javaClass = jpype.JClass('JAR.HUI')
javaInstance = javaClass()
TH=2#阈值数组[2,3,4,5,6,7,8,9]
embedding_name="onehot"
time_start=time.strftime('%H:%M:%S',time.localtime(time.time()))#开始时间
start = time.clock()#开始时间
print(embedding_name)
if embedding_name == 'onehot':
    embedding_dimension=128
else:
    embedding_dimension=300
            
print("数据集名称: {}".format(dataset_name))
print("最原始的训练数据大小: {:d}".format(int(len(x_text)*(1-test_percentage))))
fRreult="算法:HUNN"+"\n"
fRreult=fRreult+"数据集:"+dataset_name+"\n"
fRreult=fRreult+"词向量:"+embedding_name+"\n"
fRreult=fRreult+"TH:"+str(TH)+"\n"
x,vocab_processor=Vocab_proces(x_text)
voc_len=len(vocab_processor.vocabulary_)
x_train, x_dev, y_train, y_dev=DATA_split(x,y,test_percentage)

#需要将数据分为训练集和测试集，然后将训练集进行处理，而测试集不变，对x_train和y_train输入到JAR中进行处理
#如果使用高效用词集过虑,则需要删除全为0的事务和标签
x_train, y_train=HUI_DEL_Zero(x_train, y_train)
print("删除x_train,y_train的全0元素后的数据: {:d}".format(len(x_train)))
trans, quantity, label = Text_To_HUI(x_train, y_train)#得到HUI的输入
print("调用jar包，挖掘HUI……")
# 获取经过高效用项集过虑后的数据,获取成对词数据
result= javaInstance.GetHUI(TH,Filter_Mode, trans, quantity, label)
x_train, y_train, x_dev, y_dev = HUI_To_Text(result,x_dev,y_dev,Test_Filter)
print("经HUI过虑后的x_train和y_train数据量:{:d}, {:d}".format(len(x_dev),len(x_train)))
print("训练/验证比率: {:d}/{:d}".format(len(y_train), len(y_dev)))#训练/验证比率: 9596/1066
MAXacc=0.0;#最高准确率

with tf.Graph().as_default():
    #这个session配置，按照前面的gpu，cpu自动选择
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)#建立一个配置(如GPU)如上的会话
    with sess.as_default():#设置会话
        nn = TextHUNN(sequence_length=x_train.shape[1],num_classes=y_train.shape[1],vocab_size=voc_len,
            embedding_size=embedding_dimension, batch_size=batch_size,l2_reg_lambda=l2_reg_lambda)
        #定义一个变量step，从0开始，每次加1
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # 定义优化器，里面是学习速率，选择优化算法，建立优化器
        optimizer = tf.train.AdamOptimizer(nn.learning_rate)
        #选择目标函数，计算梯度；返回的是梯度和变量
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(nn.loss, tvars), grad_clip)
        # grads_and_vars = optimizer.compute_gradients(nn.loss)
        grads_and_vars = tuple(zip(grads, tvars))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)#运用梯度
        
        grad_summaries = []# 跟踪梯度值和稀疏度（可选），暂时不懂
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
 
        time_addr=time.strftime('%Y%m%d_%H_%M_%S',time.localtime(time.time()))#以时间为文件夹
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", time_addr))#输出模型地址
        print("模型写入到 {}\n".format(out_dir))#
         # 损失函数和准确率的参数保存
        loss_summary = tf.summary.scalar("loss", nn.loss)
        acc_summary = tf.summary.scalar("accuracy", nn.accuracy)
        # 训练数据保存
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        # 测试数据保存
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        # 检查点目录。 Tensorflow假设此目录已存在，因此我们需要创建它
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))#创建checkpoints地址
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")#创建model地址
        if not os.path.exists(checkpoint_dir):#True
            os.makedirs(checkpoint_dir)#如果为False 则创建文件地址
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)#用于恢得模型的参数
        vocab_processor.save(os.path.join(out_dir, "vocab"))#写入词典
        sess.run(tf.global_variables_initializer())#初始化所有全局变量
        #定义了一个训练函数，输入为1个batch
        def train_step(x_batch, y_batch, learning_rate):
            feed_dict = {
                nn.input_x: x_batch,
                nn.input_y: y_batch,
                nn.dropout_keep_prob: dropout_keep_prob,#0.5,退出保持概率
                nn.learning_rate: learning_rate
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, nn.loss, nn.accuracy],
                feed_dict)
            #time_str = datetime.datetime.now().isoformat()
            #print("{}: step {}, loss {:g}, acc {:g}, lr {:g}".format(time_str, step, loss, accuracy, learning_rate))
            train_summary_writer.add_summary(summaries, step)
        #定义了一个测试函数，用于验证集，输入为一个batch
        #验证集太大，会爆内存，采用batch的思想进行计算，下面生成多个子验证集
        def dev_step(x_batch, y_batch, writer=None):
            loss_sum = 0
            accuracy_sum = 0
            summaries = None
            step = None
            batches_in_dev = len(y_batch) // batch_size
            for batch in range(batches_in_dev):
                start_index = batch * batch_size
                end_index = (batch + 1) * batch_size
                feed_dict = {
                    nn.input_x: x_batch[start_index:end_index],
                    nn.input_y: y_batch[start_index:end_index],
                    nn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, nn.loss, nn.accuracy],
                    feed_dict)
                loss_sum += loss
                accuracy_sum += accuracy
                if writer:
                    writer.add_summary(summaries, step)
            loss = loss_sum / batches_in_dev
            accuracy = accuracy_sum / batches_in_dev
            time_str = datetime.datetime.now().isoformat()# 获取当前时间'2018-08-09T14:40:43.627831'
            print("{}: 步 {}, 损失 {:g}, 准确率 {:g}".format(time_str, step, loss, accuracy))
            global MAXacc
            if accuracy>MAXacc:
                MAXacc=accuracy
            if writer:
                writer.add_summary(summaries, step)
        # 产生batches，batch_size=64, num_epochs=60，开始训练
        batches = batch_iter(
            list(zip(x_train, y_train)), batch_size, num_epochs)
        #使用动态学习率，值越高训练速度越快
        max_learning_rate = 0.01
        min_learning_rate = 0.001
        decay_speed = decay_coefficient*len(y_train)/batch_size
        
        # 训练步骤，对于每一个batch...
        counter = 0
        for batch in batches:# 训练 loop. 对于每一个batch...
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_speed)
            counter += 1
            x_batch, y_batch = zip(*batch)# 按batch把数据拿进来
            train_step(x_batch, y_batch, learning_rate)

            current_step = tf.train.global_step(sess, global_step)# 第几步
            if current_step % evaluate_every == 0:# 每200执行一次测试
                print("\n实验结果:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % checkpoint_every == 0:# 每200次执行一次保存模型
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)# 定义模型保存路径
                #print("保存模型检查点到 {}\n".format(path))

        print("最高准确率 {:g}".format(MAXacc))
        time_end=time.strftime('%H:%M:%S',time.localtime(time.time()))#结束时间
        end = time.clock()
        TimeConsuming=round((end-start)/60)
        print("开始 {}, 结束 {}, 耗时 {}分".format(time_start,time_end,TimeConsuming))
        
        f2="D:\\RESULT\\"+dataset_name+"_HUNN_"+str(TH)+"_"+time_addr+".txt"
        fRreult=fRreult+"开始:"+str(time_start)+" 结束:"+str(time_end)+" 耗时:"+str(TimeConsuming)+"分,最高准确率:"+str(MAXacc)
        #将结果数据写入到文件中，以时间命名
        with open(f2,'w') as f2:
            f2.writelines(fRreult)
            
jpype.shutdownJVM()    
