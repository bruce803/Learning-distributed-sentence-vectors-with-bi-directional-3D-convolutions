import torch.nn as nn
import torch.nn.functional as F
import copy
import torch

class NetDBPediaMaxOverTime3(nn.Module): #acc %, loss: , kernel_conv1=80, learn_rate=0.0001
    def __init__(self, dropout_p=0.4):
        super(NetDBPediaMaxOverTime3, self).__init__()

        # Define parameters
        self.dropout_p = dropout_p
        # self.max_length = 140  # L
        self.training = True

        self.conv1 = nn.Conv3d(1, 50, (20, 20, 3), stride=(1, 1, 1))
        self.pool1d = nn.MaxPool1d(3, 3, dilation=2)
        self.fc1 = nn.Linear(50*25, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 15)


    #         self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv1(x)
        x = x.squeeze(2)#squezze降维
        x = x.squeeze(2)  # (batch, kernel, words)
        x = F.relu(self.pool1d(x))
        # print(x.shape)
        x = x.view(-1, 50*25)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x
        
        
class NetDBPediaMaxOverTime3flip(nn.Module): #acc %, loss: , kernel_conv1=80, learn_rate=0.0001
    def __init__(self, dropout_p=0.4):
        super(NetDBPediaMaxOverTime3flip, self).__init__()

        # Define parameters
        self.dropout_p = dropout_p
        # self.max_length = 140  # L
        self.training = True

        self.conv1 = nn.Conv3d(1, 50, (20, 20, 3), stride=(1, 1, 1))
        self.pool1d = nn.MaxPool1d(3, 3, dilation=2)
        self.fc1 = nn.Linear(100*25, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 15)


    #         self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
    
        x_raw = copy.deepcopy(x) #  x before copy shape: torch.Size([400, 1, 20, 20, 200])
        x = self.conv1(x) # x after copy shape: torch.Size([400, 50, 1, 1, 198])
        x = x.squeeze(2)
        x = x.squeeze(2)  # (batch, kernel, words)
        x = F.relu(self.pool1d(x))
        # print(x.shape)
        x1 = x_raw.flip(-1)
        x1 = self.conv1(x1)
        x1 = x1.squeeze(2)
        x1 = x1.squeeze(2)  # (batch, kernel, words)
        x1 = F.relu(self.pool1d(x1))
        # print(x1.shape)
        
        x = torch.cat([x,x1],1)

        x = x.view(-1, 100*25)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x
        
class NetDBPediaMaxOverTime3flip_tsne(nn.Module): #acc %, loss: , kernel_conv1=80, learn_rate=0.0001
    def __init__(self, dropout_p=0.4):
        super(NetDBPediaMaxOverTime3flip_tsne, self).__init__()

        # Define parameters
        self.dropout_p = dropout_p
        # self.max_length = 140  # L
        self.training = True

        self.conv1 = nn.Conv3d(1, 50, (20, 20, 3), stride=(1, 1, 1))
        self.pool1d = nn.MaxPool1d(3, 3, dilation=2)
        self.fc1 = nn.Linear(100*25, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 15)


    #         self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
    
        x_raw = copy.deepcopy(x) #  x before copy shape: torch.Size([400, 1, 20, 20, 200])
        x = self.conv1(x) # x after copy shape: torch.Size([400, 50, 1, 1, 198])
        x = x.squeeze(2)
        x = x.squeeze(2)  # (batch, kernel, words)
        x = F.relu(self.pool1d(x))
        x3=x.view(-1,50*25)
        #print("x3.shape=",x3.shape)
        
        x1 = x_raw.flip(-1)
        x1 = self.conv1(x1)
        x1 = x1.squeeze(2)
        x1 = x1.squeeze(2)  # (batch, kernel, words)
        x1 = F.relu(self.pool1d(x1))
        x3flip=x1.view(-1,50*25)
        #print("x3flip.shape=",x3flip.shape)
        
        x = torch.cat([x,x1],1)

        x = x.view(-1, 100*25)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x,x3,x3flip

class NetDBPediaMaxOverTime2345(nn.Module): #acc %, loss: , kernel_conv1=80, learn_rate=0.0001
    def __init__(self, dropout_p=0.4):
        super(NetDBPediaMaxOverTime2345, self).__init__()

        # Define parameters
        self.dropout_p = dropout_p
        # self.max_length = 140  # L
        self.training = True

        self.conv1 = nn.Conv3d(1, 50, (20, 20, 2), stride=(1, 1, 1))
        self.conv2 = nn.Conv3d(1, 50, (20, 20, 3), stride=(1, 1, 1))
        self.conv3 = nn.Conv3d(1, 50, (20, 20, 4), stride=(1, 1, 1))
        self.conv4 = nn.Conv3d(1, 50, (20, 20, 5), stride=(1, 1, 1))
        self.pool1d = nn.MaxPool1d(3, 3, dilation=2)
        self.fc1 = nn.Linear(50*99, 700)
        self.fc2 = nn.Linear(700, 120)
        self.fc3 = nn.Linear(120, 15)
    #         self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x_raw1 = copy.deepcopy(x)
        x_raw2 = copy.deepcopy(x)
        x_raw3 = copy.deepcopy(x)

        x1 = self.conv1(x)
        x1 = x1.squeeze(2)
        x1 = x1.squeeze(2)  # (batch, kernel, words)
        x1 = F.relu(self.pool1d(x1))

        x2 = self.conv2(x_raw1)
        x2 = x2.squeeze(2)
        x2 = x2.squeeze(2)  # (batch, kernel, words)
        x2 = F.relu(self.pool1d(x2))

        x3 = self.conv3(x_raw2)
        x3 = x3.squeeze(2)
        x3 = x3.squeeze(2)  # (batch, kernel, words)
        x3 = F.relu(self.pool1d(x3))

        x4 = self.conv4(x_raw3)
        x4 = x4.squeeze(2)
        x4 = x4.squeeze(2)  # (batch, kernel, words)
        x4 = F.relu(self.pool1d(x4))

        x = torch.cat((x1,x2,x3,x4),2)

        x = x.view(-1, 50 * 99)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)

        return x
class NetDBPediaMaxOverTime2345flip(nn.Module): #acc %, loss: , kernel_conv1=80, learn_rate=0.0001
    def __init__(self, dropout_p=0.4):
        super(NetDBPediaMaxOverTime2345flip, self).__init__()

        self.dropout_p = dropout_p
        self.training = True

        self.conv1 = nn.Conv3d(1, 50, (20, 20, 2), stride=(1, 1, 1))
        self.conv2 = nn.Conv3d(1, 50, (20, 20, 3), stride=(1, 1, 1))
        self.conv3 = nn.Conv3d(1, 50, (20, 20, 4), stride=(1, 1, 1))
        self.conv4 = nn.Conv3d(1, 50, (20, 20, 5), stride=(1, 1, 1))
        self.pool1d = nn.MaxPool1d(3, 3, dilation=2)
        self.fc1 = nn.Linear(100*99, 1100)
        self.fc2 = nn.Linear(1100, 120)
        self.fc3 = nn.Linear(120, 15)

    def forward(self, x):
        x_raw2 = copy.deepcopy(x)
        x_raw3 = copy.deepcopy(x)
        x_raw4 = copy.deepcopy(x)
        x_raw5 = copy.deepcopy(x)
        x_raw22 = copy.deepcopy(x)
        x_raw32 = copy.deepcopy(x)
        x_raw42 = copy.deepcopy(x)
        x_raw52 = copy.deepcopy(x)

        x2 = self.conv1(x)
        x2 = x2.squeeze(2)
        x2 = x2.squeeze(2)
        x2 = F.relu(self.pool1d(x2))
        x_raw2 = x_raw2.flip(-1)
        x2_flip = self.conv1(x_raw2)#flip翻转，反向传播
        x2_flip = x2_flip.squeeze(2)
        x2_flip = x2_flip.squeeze(2)
        x2_flip = F.relu(self.pool1d(x2_flip))        
        x2 = torch.cat([x2,x2_flip],1)

        x3 = self.conv2(x_raw3)
        x3 = x3.squeeze(2)
        x3 = x3.squeeze(2)
        x3 = F.relu(self.pool1d(x3))
        x_raw32 = x_raw32.flip(-1)
        x3_flip = self.conv2(x_raw32)
        x3_flip = x3_flip.squeeze(2)
        x3_flip = x3_flip.squeeze(2)
        x3_flip = F.relu(self.pool1d(x3_flip))        
        x3 = torch.cat([x3,x3_flip],1)#concatnate拼接，按维数1拼接（横着拼）

        x4 = self.conv3(x_raw4)
        x4 = x4.squeeze(2)
        x4 = x4.squeeze(2)
        x4 = F.relu(self.pool1d(x4))
        x_raw42 = x_raw42.flip(-1)
        x4_flip = self.conv3(x_raw42)
        x4_flip = x4_flip.squeeze(2)
        x4_flip = x4_flip.squeeze(2)
        x4_flip = F.relu(self.pool1d(x4_flip))        
        x4 = torch.cat([x4,x4_flip],1)
        
        x5 = self.conv4(x_raw5)
        x5 = x5.squeeze(2)
        x5 = x5.squeeze(2)
        x5 = F.relu(self.pool1d(x5))
        x_raw52 = x_raw52.flip(-1)
        x5_flip = self.conv4(x_raw52)
        x5_flip = x5_flip.squeeze(2)
        x5_flip = x5_flip.squeeze(2)
        x5_flip = F.relu(self.pool1d(x5_flip))        
        x5 = torch.cat([x5,x5_flip],1)
        
        x = torch.cat((x2,x3,x4,x5),2)

        x = x.view(-1, 100 * 99)#改变tensor形状，-1位置为让电脑替我们计算的维度
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)

        return x
