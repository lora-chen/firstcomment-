import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    # class torch.nn.Module
    # 官方文档
    # 所有网络的基类
    # 你的模型也应该继承这个类。

    #   Model description
    #   model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim,hidden_dim=hidden_dim,
    #                        vocab_size=len(corpus.dictionary),label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()   # _init__()确保父类被正确的初始化了：

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # create model, regulate size
        # Input should be module containing 23590 tensors of size 100
        # 模块的输入是一个下标的列表，输出是对应的词嵌入。

        # 官方文档
        # class torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        # 参数：
        # num_embeddings(int) - 嵌入字典的大小
        # embedding_dim(int) - 每个嵌入向量的大小
        # padding_idx(int, optional) - 如果提供的话，输出遇到此下标时用零填充
        # max_norm(float, optional) - 如果提供的话，会重新归一化词嵌入，使它们的范数小于提供的值
        # norm_type(float, optional) - 对于max_norm选项计算p范数时的p
        # scale_grad_by_freq(boolean, optional) - 如果提供的话，会根据字典中单词频率缩放梯度
        # 变量：
        # weight(Tensor) - 形状为(num_embeddings, embedding_dim)的模块中可学习的权值
        # 形状：
        # 输入： LongTensor(N, W), N = mini - batch, W = 每个mini - batch中提取的下标数
        # 输出： (N, W, embedding_dim)



        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # Applies a multi - layer long short - term memory(LSTM) RNN to an input sequence.
        # input layer size is 100, hidden layer size is 50
        # 层数为单层可能是这里的  num_layer 没有设定    ？？？？？？？？？？？？？

        # 官方文档 https://pytorch.org/docs/stable/nn.html
        # 参数说明:
        # input_size – 输入的特征维度
        # hidden_size – 隐状态的特征维度
        # num_layers – 层数（和时序展开要区分开）
        # bias – 如果为False，那么LSTM将不会使用bias weights b_ih and b_hh 默认为True。
        # batch_first – 如果为True，那么输入和输出Tensor的形状为(batch, seq, feature)
        # dropout – 如果非零的话，将会在RNN的输出上加个dropout，最后一层除外。
        # bidirectional – 如果为True，将会变成一个双向RNN，默认为False。


        self.hidden2label = nn.Linear(hidden_dim, label_size)
        # x1 = nn.Linear(hidden_dim, label_size).weight.shape   torch.Size([8, 50])
        # x2 = nn.Linear(hidden_dim, label_size).bias.shape      torch.Size([8])

        # 输入应该是一个  什么 * 50 的torch

        # 有点难理解这里的数据结构 ？？？？？？？？？？？？？？？？？？？

        # Create linear layer
        # Applies a linear transformation to the incoming data: y =  xA ^ T + b, x is a matrix

        # in_features - 每个输入样本的大小
        # out_features - 每个输出样本的大小
        # bias - 若设置为False，这层不会学习偏置。默认值：True
        # 形状：
        # 输入:
        # vector(N, in_features)
        # vector(N, in_features)
        # 输出：
        # (N, out_features)
        # (N, out_features)
        # 变量：
        # weight - 形状为(out_features x in_features)的模块中可学习的权值
        # bias - 形状为(out_features) 的模块中可学习的偏置

        self.hidden = self.init_hidden()  # 返回保存着batch中每个元素的初始化隐状态的Tensor
            # 返回batch中每个元素的初始化细胞状态的Tensor

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            # 在Torch中的Variable就是一个存放会变化的值的地理位置.里面的值会不停的变化.值是Tensor如果用一个 Variable进行计算, 那返回的也是一个同类型的
            # Variable
            # Create two size 5* 50 tensors, filling with 0

            # 一开始那个1意味着小数点后一位
            # print(torch.zeros(1, 3, 5))
            # tensor([[[0., 0., 0., 0., 0.],
            #          [0., 0., 0., 0., 0.],
            #         [0., 0., 0., 0., 0.]]])

            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))

            # LSTM输入: input, (h_0, c_0)
            # input(seq_len, batch, input_size): 包含输入序列特征的Tensor。也可以是packed variable ，详见[pack_padded_sequence](
            #  torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False[source])
            #  h_0(num_layers * num_directions, batch, hidden_size): 保存着batch中每个元素的初始化隐状态的Tensor
            # c_0(num_layers * num_directions, batch, hidden_size): 保存着batch中每个元素的初始化细胞状态的Tensor
        return (h0, c0)

    def forward(self, sentence):
        # 之前创建的Embedding层是用VOCAB SIZE，然后再FORWARD里面用的是SENTENCE Size， 有点搞不懂 ？？？？？？？？？
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # create model, regulate size
        # Sentence should be module containing 23590 tensors of size 100
        #输出： (N, W, embedding_dim

        embeds = self.word_embeddings(sentence)
        # Input : Sentence tensor size 32 *5    print (sentence.size()) torch.Size([32, 5])
        # Output: Embeds tensor 32* 5* 10       print (embeds.size())  torch.Size([32, 5, 100])

        # x = embeds
        x = embeds.view(len(sentence), self.batch_size, -1)
        #直接使用也是一样的
        # 好像本来就是32，5，100的tensor，转化结果前后应该是一样的 ？？？？？？？？？？？？

        # Input : Embeds.torch.Size([32, 5, 100])
        # Output: x.torch.Size([32,5,100])   print (x.shape) torch.Size([32, 5, 100])

        # -1 means no sure about the size of one row
        # View() method can regroup the tensor into different size , but does not change content.
        # e.g. a = torch.arange(1, 17)  # a's shape is (16,)
        # a.view(4, 4) # output below
        #   1   2   3   4
        #   5   6   7   8
        #   9  10  11  12
        #  13  14  15  16
        # [torch.FloatTensor of size 4x4]

        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # 右边相当于把X作为输入，h0和c0作为初始状态变量  == input, (h_0, c_0)  h0和 c0 是两个5* 50 tensors，初始值都为0.0，然后不断迭代
        # 怎么看出循环过程 ？？？？？？？？？？？
        # 左边是LSTM输出 output, (h_n, c_n)
        # output(seq_len, batch, hidden_size * num_directions):
        #                 保存RNN最后一层的输出的Tensor。
        #                 如果输入是torch.nn.utils.rnn.PackedSequence，那么输出也是torch.nn.utils.rnn.PackedSequence。
        # h_n(num_layers * num_directions, batch, hidden_size): Tensor，保存着RNN最后一个时间步的隐状态。
        # c_n(num_layers * num_directions, batch, hidden_size): Tensor，保存着RNN最后一个时间步的细胞状态


        y  = self.hidden2label(lstm_out[-1])
        # Input: The last output of LSTM_Out  (-1 means last output)  print ( lstm_out[-1].shape)---torch.Size([5, 50])
        # Output:  print( y.shape)  一开始都是torch.Size([5, 8])，最后一个是 torch.Size([4, 8]) ????????????????????????
        # 在 t 时刻，LSTM 的输入有三个：当前时刻网络的输入值 x_t、上一时刻 LSTM 的输出值 h_t-1、以及上一时刻的单元状态 c_t-1；
        # LSTM 的输出有两个：当前时刻 LSTM 输出值 h_t、和当前时刻的单元状态 c_t. 应该是对应着5个状态和8个label的得分，然后最后一次结束有一个状态没有了，所以是4，8 ？？？？？？
        # X的值最后是torch.Size([32, 4, 100])

        # print (x.shape)
        # print("This is X")
        # print(len(sentence))
        # print("This is len(sentence)")
        # print(lstm_out[-1].shape)
        # print("This is Lstm_out[-1]")
        # print(y.shape)
        # print ("This is Y")
        return y   #最后返回的应该是一个4.8的tensor

    # 编写前向过程
    # '''def forward(self, inputs):
    #     embeds = self.embeddings(inputs).view((1, -1))  # Input is voc vecture and project to Embed layer
    #     out = F.relu(self.linear1(embeds))              # Calculate Hidden layer output, Relu activation function
    #     out = self.linear2(out)                         # Self.linear2 is output layer
    #     log_probs = F.log_softmax(out)                  # Calculate  forward process
    #  ''''''return log_probs
    #
    # # 第二个前向 
    # 预处理文本转成稠密向量
    #         embeds=self.embedding((inputs))
    #         #根据文本的稠密向量训练网络
    #         out,self.hidden=self.lstm(embeds.view(len(inputs),1,-1),self.hidden)
    #         #做出预测
    #         tag_space=self.out2tag(out.view(len(inputs),-1))
    #         tags=F.log_softmax(tag_space,dim=1)
    #         return tags
