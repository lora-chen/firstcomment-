import os
import torch
import copy
from torch.utils.data import DataLoader
import utils.DataProcessing as DP
import utils.LSTMClassifier as LSTMC
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

use_plot = True          # Give the tag that whether you want to save data and plot the figure
use_save = True          # Give the tag that whether save
# if use_plot:
#   ....
# if use_save:

if use_save:             # If Yes, import the serialize package and data time
    import pickle
    #序列化用的包
    from datetime import datetime

# Store path ..\LSTM-Classification-Pytorch\data\test_txt\1.txt
#            ..\LSTM-Classification-Pytorch\data\train_txt
DATA_DIR = 'data'                         # ..\LSTM-Classification-Pytorch\    $ DATA_DIR      \test_txt\1.txt
TRAIN_DIR = 'train_txt'                   # ..\LSTM-Classification-Pytorch\data\   $TRAIN_DIR  \1.txt
TEST_DIR = 'test_txt'                     # Same like above

# Train file list document  ..LSTM-Classification-Pytorch\data\train_txt.txt
# Test file list document  ..LSTM-Classification-Pytorch\data\test_txt.txt
# For each file, store the l;ist of document. Eg. 1.txt|2.txt|3.txt....
TRAIN_FILE = 'train_txt.txt'
TEST_FILE = 'test_txt.txt'

# Training data label file list document  ..LSTM-Classification-Pytorch\data\train_label.txt
# Store the list of training data label  e.g. 1|0|2| 3| 5  , project to train_txt.txt
# Supervisor Learning
TRAIN_LABEL = 'train_label.txt'
TEST_LABEL = 'test_label.txt'


## parameter setting
epochs = 10                          # Epoch of learning processing
                                        # Save time set epochs as 10
                                        # Epochs = 50
batch_size = 5                         # Batch Gradient decrease
use_gpu = torch.cuda.is_available()    # IF use CUDA
                                       # Will use following:
                                       # if use_gpu:
                                       # model = model.cuda()
learning_rate = 0.01                   # Initialize learning rate

# model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim,hidden_dim=hidden_dim,
#        vocab_size=len(corpus.dictionary),label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))     # New learning rate = learning rate * (0.1 ^ (Epoch // 10))  Adative learning rate algorithm
    for param_group in optimizer.param_groups:      # optimizer通过param_group来管理参数组.param_group中保存了参数组及其对应的学习率,动量等等.
                                                    # 可以通过更改param_group[‘lr’]的值来更改对应参数组的学习率
        param_group['lr'] = lr
    return optimizer


if __name__=='__main__':
    ### parameter setting
    embedding_dim = 100
    hidden_dim = 50
    sentence_len = 32

    # Store path ..\LSTM-Classification-Pytorch\data\train_txt
    train_file = os.path.join(DATA_DIR, TRAIN_FILE)    # joint connect the path    String  'train_txt.txt'
    test_file = os.path.join(DATA_DIR, TEST_FILE)      # 'data\\test_txt.txt'
    fp_train = open(train_file, "r")                   # Only read document
                                                       # Offer the index for train dacument 1.txt
                                                       # 1.txt
                                                       # 2.txt
                                                       # 3.txt
    '''     Character   Meaning
  'r'     open for reading (default)
  'w'     open for writing, truncating the file first
  'x'     open for exclusive creation, failing if the file already exists
  'a'     open for writing, appending to the end of the file if it exists
  'b'     binary mode
  't'     text mode (default)
  '+'     open a disk file for updating (reading and writing)
  'U'     universal newlines mode (deprecated)'''

    train_filenames = [os.path.join(TRAIN_DIR, line.strip()) for line in fp_train]
    #  Read all train_filenames in variable
    #  Result:     class 'list'>: ['train_txt\\1.txt', 'train_txt\\2.txt', 'train_txt\\3.txt', 'train_txt\\4.txt', 'train_txt\\5.txt',
    #                     'train_txt\\6.txt', 'train_txt\\7.txt', 'train_txt\\8.txt', 'train_txt\\9.txt', 'train_txt\\10.txt',

    filenames = copy.deepcopy(train_filenames)  # Hard copy. "Filenames" copy a independent version of "train_filenames"
    # 0001 = {str} 'train_txt\\2.txt'
    # 0000 = {str} 'train_txt\\1.txt'
    # 0002 = {str} 'train_txt\\3.txt'
    # 0003 = {str} 'train_txt\\4.txt' ...etc
    fp_train.close()

#  close file
    # Same as above
    fp_test = open(test_file, 'r')
    test_filenames = [os.path.join(TEST_DIR, line.strip()) for line in fp_test]
    fp_test.close()
# Read test files

    filenames.extend(test_filenames)    # 7574 documents
# Now "filenames" have both test and train file name
    # 0001 = {str} 'train_txt\\2.txt'          plus  0971 = {str} 'test_txt\\1.txt'
    # 0000 = {str} 'train_txt\\1.txt'                0972 = {str} 'test_txt\\2.txt'
    # 0002 = {str}'train_txt\\3.txt'                 0973 = {str} 'test_txt\\3.txt'

    corpus = DP.Corpus(DATA_DIR, filenames)  # 这个最后创建的东西有点看不懂 ？？？？？？
    # return ids    # Tokenize Tensor
    nlabel = 8        # Label from 0 to 7

    ### create model
    model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim,hidden_dim=hidden_dim,
                           vocab_size=len(corpus.dictionary),label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)
    # len(corpus.dictionary) 23590
    # len(corpus.dictionary) --- return len(self.idx2word)
    # LSTMClassifier(
    #   (word_embeddings): Embedding(23590, 100)
    #   (lstm): LSTM(100, 50)
    #   (hidden2label): Linear(in_features=50, out_features=8, bias=True)

    if use_gpu:
        model = model.cuda()
    ### data processing
    dtrain_set = DP.TxtDatasetProcessing(DATA_DIR, TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus)
    # data type <utils.DataProcessing.TxtDatasetProcessing object at 0x000001F8149C9518>
    # 以后怎么标注这种object ？？？？？？？
    # Create training tensor

    # 读入训练文档   https://zhuanlan.zhihu.com/p/35698470 这个看看
    train_loader = DataLoader(dtrain_set,              # Input dataset
                          batch_size=batch_size,       # batch_size (int, optional): how many samples per batch to load (default: 1).
                          shuffle=True,                # set to ``True`` to have the data reshuffled at every epoch (default: False).
                          num_workers=4                # 0的话表示数据导入在主进程中进行，其他大于0的数表示通过多个进程来导入数据，可以加快数据导入速度
                         )


    dtest_set = DP.TxtDatasetProcessing(DATA_DIR, TEST_DIR, TEST_FILE, TEST_LABEL, sentence_len, corpus)
        # Create testing tensor
    
    test_loader = DataLoader(dtest_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4
                             )
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)   # Optimaztion: SGD 算法
        # class torch.optim.SGD(params, lr=, momentum = 0,dampening = 0,weight_decay = 0,nesterov = False)
        # params(iterable) – 待优化参数的iterable或者是定义了参数组的dict
        # lr(float) – 学习率
        # momentum(float, 可选) – 动量因子（默认：0）
        # weight_decay(float, 可选) – 权重衰减（L2惩罚）（默认：0）
        # dampening(float, 可选) – 动量的抑制因子（默认：0）
        # nesterov(bool, 可选) – 使用Nesterov动量（默认：False）

    loss_function = nn.CrossEntropyLoss()           # 交叉熵损失函数 官方文档上有详细的公式计算，这里就不备注了

    train_loss_ = []       # 初始化训练和测试集正确率和损失
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    
    for epoch in range(epochs):
            optimizer = adjust_learning_rate(optimizer, epoch)  # 上面已经备注
    
            ## training epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0

            for iter, traindata in enumerate(train_loader):  # Create index for sequence train_loader, pick batch and shuffle every time.
                                                             # enumerate的作用就是对可迭代的数据进行标号并将其里面的数据和标号一并打印出来。
                                                             # 每一个 iter 释放一小批数据用来学习
                                                             # ??????????这里读数据怎么读的看不懂

                train_inputs, train_labels = traindata       # train_inputs torch.Size([5, 32] 5来自batch_size, 32 来自sen_len
                                                             # train_labels torch.Size([5, 1])  5来自batch_size,真实label

                #print (train_labels.shape)                   torch.Size([5, 1])
                train_labels = torch.squeeze(train_labels)   # torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行
                                                             # squeeze(a)就是将a中所有为1的维度删掉

                #print('Epoch: ', epoch, '| Step: ', iter, '| train_inputs: ',train_inputs.numpy(), '| train_labels: ', train_labels.size(), '| train_labels:.size ', train_inputs.size())
            #   Epoch:  0 | Step:  1084        train_inputs:.size  torch.Size([5, 32]          train_labels: [5 0 0 0 1]    torch.Size([5])

            if use_gpu:
                    train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()
            else: train_inputs = Variable(train_inputs)
    
            model.zero_grad()  #清空梯度缓存
           # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/3
           # 根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
           # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了。
           # 现在还不是很理解

            model.batch_size = len(train_labels)   # batch_size = 5
            model.hidden = model.init_hidden()     # model.hidden: tuple type return (h0, c0)

            # print(train_inputs.shape)  torch.Size([5, 32])
            # print(train_inputs.t().shape)    torch.Size([32, 5])
            output = model(train_inputs.t())    # Transpose train_inputs tensor and use it as input
                                                # Output torch.Size([5, 8])

            loss = loss_function(output, Variable(train_labels))  # Calculate error cross_entropy(predicted value, class )
                                                                  # 公式在官方文档里，这里不注释了



            loss.backward()      # torch.autograd.backward(variables, grad_variables=None, retain_graph=None, create_graph=False)
                                 # 这里是默认情况，相当于out.backward(torch.Tensor([1.0]))
# 给定图的叶子节点variables, 计算图中变量的梯度和。 计算图可以通过链式法则求导。如果variables中的任何一个variable是 非标量(non-scalar)的，且requires_grad=True。
# 那么此函数需要指定grad_variables，它的长度应该和variables的长度匹配，里面保存了相关variable的梯度(对于不需要gradient tensor的variable，None是可取的)。
# 此函数累积leaf variables计算的梯度。你可能需要在调用此函数之前将leaf variable的梯度置零。
# 参数：
#
# variables（变量的序列） - 被求微分的叶子节点，即 ys 。
# grad_variables（（张量，变量）的序列或无） - 对应variable的梯度。仅当variable不是标量且需要求梯度的时候使用。
# retain_graph（bool，可选） - 如果为False，则用于释放计算grad的图。请注意，在几乎所有情况下，没有必要将此选项设置为True，通常可以以更有效的方式解决。默认值为create_graph的值。
# create_graph（bool，可选） - 如果为True，则将构造派生图，允许计算更高阶的派生产品。默认为False

            # 更新的三步：
            1.
            # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            # 这一句代码中optimizer获取到了所有parameters的引用，每个parameter都包含梯度（gradient），optimizer可以把梯度应用上去更新parameter。
            2.
            # loss = loss_function(output,Variable(train_labels))
            # prediction和true class之间进行比对（熵或者其他lossfunction），产生最初的梯度
            # loss.backward()
            # 反向传播到整个网络的所有链路和节点。节点与节点之间有联系，因此可以反向链式传播梯度
         # 3.
            optimizer.step()
            # apply所有的梯度以更新parameter的值.因为step（）更新所有参数，所以不用指明梯度


            _, predicted = torch.max(output.data, 1) #  返回每一行中最大值的那个元素，且返回其索引
                                                     #  Predicted 前面那个逗号是为了返回索引，而不是具体的值，但是具体怎么看代码不知道 ？？？？？？？
                                                     #  输入：The size of output.data torch.Size([5, 8]),
                                                     #  输出：predicted  torch.Size([5])
            # train_loss_ = []  # 初始化训练和测试集正确率和损失
            # test_loss_ = []
            # train_acc_ = []
            # test_acc_ = []

            total_acc += (predicted == train_labels).sum()  # 多少个训练对了，是个size 0的tensor，要用。item（）来看
            # print (total_acc.item())
            total += len(train_labels)          # len(train_labels) = 5, 每次训练一次加5
            total_loss += loss.item()           # 需要额外加上.item() 来获得里面的值
            train_loss_.append(total_loss / total)  # 每做一次加一次
            train_acc_.append(total_acc / total)

            # 注释到这


            ## testing epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            for iter, testdata in enumerate(test_loader):
                test_inputs, test_labels = testdata
                test_labels = torch.squeeze(test_labels)
    
                if use_gpu:
                    test_inputs, test_labels = Variable(test_inputs.cuda()), test_labels.cuda()
                else: test_inputs = Variable(test_inputs)
    
    
                model.batch_size = len(test_labels)
                model.hidden = model.init_hidden()
                output = model(test_inputs.t())
    
                loss = loss_function(output, Variable(test_labels))
    
                # calc testing acc
                _, predicted = torch.max(output.data, 1)
                total_acc += (predicted == test_labels).sum()
                total += len(test_labels)
                total_loss += loss.item()
            test_loss_.append(total_loss / total)
            test_acc_.append(total_acc / total)
    
            print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
                  % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))
    
    param = {}
    param['lr'] = learning_rate
    param['batch size'] = batch_size
    param['embedding dim'] = embedding_dim
    param['hidden dim'] = hidden_dim
    param['sentence len'] = sentence_len
    
    result = {}
    result['train loss'] = train_loss_
    result['test loss'] = test_loss_
    result['train acc'] = train_acc_
    result['test acc'] = test_acc_
    result['param'] = param
    
    if use_plot:
            import PlotFigure as PF
            PF.PlotFigure(result, use_save)
    if use_save:
            filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
            result['filename'] = filename
    
            fp = open(filename, 'wb')
            pickle.dump(result, fp)
            fp.close()
            print('File %s is saved.' % filename)
    
    
    #
    # 1）注释每一句代码，不清楚的标记上，开会讨论
    # 2）注释的本质就是讲每个函数的输入、输出是什么，最好结合debug，看内存中的数据形式，比如输入是3x5的矩阵，代表了xxx
    # 3）不要花费大量时间看教程和视频，以这个代码为主，代码有什么问题，针对性的搜什么问题
    # 4）确实有困难，及时回复，初学者有困难很正常，不用自己憋着等很久，不好意思问
    # 5）建议做笔记，记录自己的经验教训
    # 6）建议注册github账号，把代码放在github上