from __future__ import print_function, division
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score,f1_score,precision_score
from sklearn import utils
import warnings,time
from sklearn.model_selection import KFold
import math
warnings.filterwarnings('ignore')

num_feats = ['duration','src_bytes','dst_bytes','wrong_fragment',
            'urgent','hot','num_failed_logins','num_compromised','su_attempted','num_root',
           'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
           'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
            'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
            'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
           'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',]

index_feats = ['protocol_type','service','flag','land','logged_in','root_shell','is_host_login','is_guest_login',]



class transData(Dataset):
    def __init__(self,index_file,raw_file,):
        self.data = pd.read_csv(raw_file)
        self.index = pd.read_csv(index_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        Xi = []
        Xv = []
        raw_data = dict(self.data.iloc[idx])
        index = dict(self.index.iloc[idx])
        label = int(raw_data['label'])
        for feat in index_feats:
            Xi.append(index[feat])
            Xv.append(1)
        for feat in num_feats:
            Xi.append(0)
            Xv.append(raw_data[feat])

        return np.array(Xi),np.array(Xv).astype(np.float32),label

def get_data_index(value_path,store_path):
    value = pd.read_csv(value_path)
    for feat in value.columns:
        if feat in num_feats:
            value[feat] = 0

    value.to_csv(store_path,index=False)


class dnn(nn.Module):
    def __init__(self, field_size, feature_sizes, k, num_class,
                 lr = 0.001,
                 batch_size = 300,
                 epoches = 100,
                 optimer='adam',
                 weight_decay = 0.0,
                 deep_layers=[300,150],
                 random_seed=999):
        super(dnn, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.k = k
        self.deep_layers = deep_layers
        self.num_class = num_class

        self.lr = lr
        self.batch_size = batch_size
        self.epoches = epoches
        self.optimer = optimer
        self.weight_decay = weight_decay
        self.random_seed = random_seed

        self.embeddings_1 = nn.ModuleList([nn.Embedding(size,1) for size in self.feature_sizes])
        self.embeddings_2 = nn.ModuleList(
            [nn.ModuleList([nn.Embedding(feature_size, self.k)
                            for _ in range(self.field_size)])
             for feature_size in self.feature_sizes])

        input_size_1 = int(self.field_size*(self.field_size-1)/2)


        self.linear_1 = nn.Linear(input_size_1,self.deep_layers[0])
        self.linear_2 = nn.Linear(self.deep_layers[0],self.deep_layers[1])
        input_size_2 = int(self.field_size + self.deep_layers[-1])
        self.linear = nn.Linear(input_size_2,self.num_class)

        torch.manual_seed(self.random_seed)

    def forward(self, Xi, Xv):

        emb_1 = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                              enumerate(self.embeddings_1)]
        emb_1 = [torch.sum(emb,1) for emb in emb_1]
        emb_1 = torch.cat([emb.unsqueeze(1) for emb in emb_1],1)

        emb_2 = [[(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for emb in f_embs]
                 for i, f_embs in enumerate(self.embeddings_2)]
        emb_2_arr = []
        for i in range(self.field_size):
            for j in range(i + 1, self.field_size):
                emb_2_arr.append(torch.sum(emb_2[i][j] * emb_2[j][i], 1).view(-1, 1))
        emb_2 = torch.cat(emb_2_arr, 1)

        deep_emb = emb_2
        linear_1 = self.linear_1(deep_emb)
        linear_1 =  F.relu(linear_1)
        linear_2 = self.linear_2(linear_1)
        linear_2 = F.relu(linear_2)
        linear_2 = torch.cat([emb_1,linear_2],1)
        out = self.linear(linear_2)
        _,label  = torch.max(out.data,1)
        return out,label


    def load_data_(self,index_path,value_path):
        index_data = pd.read_csv(index_path)
        value_data = pd.read_csv(value_path)
        label = value_data.pop('label')
        index_data = np.array(index_data).reshape((-1,self.field_size,1))
        value_data = np.array(value_data)
        label = np.array(label)

        return index_data,value_data,label

    def data_loader(self,index_path,value_path,shuffle=False):
        data = transData(index_path,value_path)
        data = DataLoader(data,batch_size=self.batch_size,shuffle=shuffle)

        return data

    def fit(self,index_path,value_path,
            test_index_path=None,
            test_value_path=None,
            shuffle = False,
            print_params = False,
            save_name = None,
            verbose = 0,
            early_stopping = 0):
        if isinstance(index_path,pd.DataFrame) and isinstance(value_path,pd.DataFrame):
            train_index, train_value, = index_path, value_path
            all_label = train_value.pop('label')
            train_index = np.array(train_index).reshape((-1, self.field_size, 1))
            train_value = np.array(train_value)
            all_label = np.array(all_label)
        else:
            train_index,train_value,all_label = self.load_data_(index_path,value_path)
        assert train_index.shape[0] == train_value.shape[0]
        length = train_index.shape[0]
        chunk = length // self.batch_size
        max_acc = 0.0
        max_f1 = 0.0
        max_epoch = 0
        model = self.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, )
        if self.optimer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),lr=self.lr)
        elif self.optimer == 'adamax':
            optimizer = torch.optim.Adamax(model.parameters(),lr=self.lr)

        else:
            print("select optimizer error")

        criterion = nn.CrossEntropyLoss()

        if print_params:
            self.print_params()

        if test_index_path is not None and test_value_path is not None:
            flags = True
            if isinstance(test_index_path, pd.DataFrame) and isinstance(test_value_path, pd.DataFrame):
                test_index, test_value, = test_index_path, test_value_path
                true_label = test_value.pop('label')
                test_index = np.array(test_index).reshape((-1, self.field_size, 1))
                test_value = np.array(test_value)
                true_label = np.array(true_label)
            else:
                test_index, test_value, true_label = self.load_data_(test_index_path, test_value_path)

            Xi_test = Variable(torch.LongTensor(test_index), volatile=True)
            Xv_test = Variable(torch.FloatTensor(test_value), volatile=True)

        else:
            flags = False

        for epoch in range(self.epoches):

            if early_stopping > 0:
                print("traing wills stop while no improving in {} rounds".format(early_stopping))
                if (epoch-max_epoch) > early_stopping:
                    break

            model.train()
            start = time.time()
            train_loss = 0
            train_acc = 0
            count_label = 0

            if shuffle:
                train_index, train_value, all_label = self.shuffle_data(train_index,train_value,all_label)

            for num in range(chunk):
                batch_time = time.time()
                start_index = num * self.batch_size
                end_index = (num + 1) * self.batch_size
                end_index = min(end_index,length)
                Xi = train_index[start_index:end_index,:,:]
                Xv = train_value[start_index:end_index,:]
                label = all_label[start_index:end_index]
                Xi = Variable(torch.LongTensor(Xi))
                Xv = Variable(torch.FloatTensor(Xv))
                label = Variable(torch.LongTensor(label))
                optimizer.zero_grad()
                outputs, pred = model(Xi, Xv)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                count_label += label.size(0)
                train_loss += loss.data[0]
                temp_loss = loss.data[0]
                train_acc += torch.sum(pred == label.data)
                temp_acc = (torch.sum(pred == label.data)) / label.size(0)
                end = time.time()
                if verbose > 0:
                    if num % int(verbose) == 0 and num > 0:
                        print('{'+'Epoch: {}, Iter: {}, Loss: {:.4f}, ACC: {:.4f}, Time: {:.2f}' \
                              .format(epoch+1,num,temp_loss, temp_acc,(end-batch_time)),'}')



            if flags:
                model.eval()
                _, pred = model(Xi_test, Xv_test)
                test_pred = list(pred)

                end = time.time()
                acc = accuracy_score(true_label, test_pred)
                if isinstance(acc, tuple):
                    acc = acc[0]
                f1 = f1_score(true_label, test_pred, average='weighted')
                if acc > max_acc:
                    max_acc = acc
                    max_f1 = f1
                    max_epoch = epoch

                    if save_name and max_acc > 0.80:
                            torch.save(model.state_dict(), save_name)

                print(
                    '[{}  {}] Train_Loss: {:.4f} Train_Acc: {:.4f} Test_Acc: {:.4f} Test_f1: {:.4f} Time: {:.2f}'.format(
                        epoch + 1, self.epoches, train_loss, train_acc / count_label, acc, f1, (end - start)
                    ), '\n',
                    '{' + 'Epoch: {} Max_acc: {:.4f} Max_F1: {:.4f}'.format(max_epoch + 1, max_acc, max_f1) + '}'
                )

            else:
                end = time.time()
                print(
                    '[{}  {}] Train_Loss: {:.4f} Train_Acc: {:.4f} Time: {:.2f}'.format(
                        epoch + 1, self.epoches, train_loss, train_acc / count_label, (end - start)
                    ))

        return max_acc, max_f1


    def shuffle_data(self, data1, data2,label):
        assert data1.shape[0] == data2.shape[0] == len(label)

        data1, data2, label = utils.shuffle(data1,data2,label,random_state=self.random_seed)

        return data1, data2, label

    def print_params(self):
        print("batch_size: {}, learning_rate: {}, num_classes: {}, optimer: {}, deep_layers: {} ".format(
            self.batch_size,self.lr,self.num_class,self.optimer,self.deep_layers),'weight_decay: {}'.format(
            self.weight_decay
        )
        )


    def init_weight(self):
        for layers  in  self.named_children():
            layer = layers[1]
            if isinstance(layer,nn.Linear):
                n = len(layer.weight.data)
                std = math.sqrt(2.0/n)
                nn.init.normal(layer.weight.data,0.0, std)
                nn.init.constant(layer.bias.data,0.01)

            elif isinstance(layer,nn.ModuleList):
                for i in layer:
                    if isinstance(i,nn.Embedding):
                        nn.init.xavier_normal(i.weight.data,)
                    elif isinstance(i,nn.ModuleList):
                        for j in i:
                            nn.init.xavier_normal(j.weight.data)
                    else:
                        print(layers)
            else:
                pass


    def predict(self,test_index_path,test_value_path):
        model.eval()
        test_index, test_value, true_label = self.load_data_(test_index_path, test_value_path)
        Xi = Variable(torch.LongTensor(test_index), volatile=True)
        Xv = Variable(torch.FloatTensor(test_value), volatile=True)
        _, pred = model(Xi, Xv)
        test_pred = list(pred)

        acc = accuracy_score(true_label, test_pred)
        if isinstance(acc, tuple):
            acc = acc[0]
        f1 = f1_score(true_label, test_pred, average='weighted')

        print('{'+'Test_Acc: {:.4f} Test_f1_Score: {:.4f}'.format(
            acc, f1,
        )+'}')
        print('Each label f1_score:',f1_score(true_label,test_pred,average=None))
        print('Each label prec_score:', precision_score(true_label, test_pred, average=None))

    def kflod_val(self,data_index,data_value,num,verbose=0,):
        index = pd.read_csv(data_index)
        value = pd.read_csv(data_value)
        kf = KFold(num, shuffle=True, random_state=999)
        acc_list = []
        f1_list = []
        for epoch, (train_index, test_index) in enumerate(kf.split(index), start=1):
            index_train, value_train = index.loc[train_index], value.loc[train_index]
            index_test, value_test = index.loc[test_index], value.loc[test_index]
            model = self.train()
            print("================= val {} start ==============".format(epoch))
            acc, f1 = model.fit(index_train, value_train, index_test, value_test,
                                shuffle=True,verbose=verbose)
            print("================val {} complete==============".format(epoch))
            acc_list.append(acc)
            f1_list.append(f1)

        print("acc: %.4f" % np.mean(acc_list))
        print("f1 score % .4f" % np.mean(f1_list))



def batch_training(save_name):
    feat_sizes = [3, 70, 11, 2, 2, 2, 2, 2, ] + [1 for _ in num_feats]
    train_data = transData('data/train_index.csv','data/train_normal.csv')
    train_data = DataLoader(train_data,batch_size=1000,shuffle=True)
    test_data = transData('data/test_index.csv','data/test_normal.csv')
    test_data  = DataLoader(test_data,batch_size=100,shuffle=False)
    model =  dnn(41, feat_sizes, 8, 5,)

    epoches = 100
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01,)

    criterion = nn.CrossEntropyLoss()

    max_acc = 0.0
    max_f1 = 0.0
    max_epoch = 0
    for epoch in range(epoches):
        start = time.time()
        train_loss = 0
        train_acc = 0
        count_label = 0
        model.train()
        for iter,(Xi,Xv,label) in enumerate(train_data,start=1):
            Xi = Variable(Xi.view(-1,41,1))
            Xv = Variable(Xv)
            label = Variable(label)
            optimizer.zero_grad()
            outputs,pred = model(Xi,Xv)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            count_label += label.size(0)
            train_loss += loss.data[0]
            temp_loss = loss.data[0]
            train_acc += torch.sum(pred == label.data)
            temp_acc = (torch.sum(pred == label.data)) / label.size(0)
            end = time.time()
            if iter % 200 == 0:
                print('{'+'Epoch: {}, Iter: {}, Loss: {:.4f}, ACC: {:.4f}, Time: {:.2f}' \
                      .format(epoch+1,iter,temp_loss, temp_acc,(end-start)),'}')

        test_pred = []
        true_label = []
        model.eval()
        for batch,(Xi,Xv,label) in enumerate(test_data,start=1):
            Xi = Variable(Xi.view(-1,41,1),volatile=True)
            Xv = Variable(Xv,volatile=True)
            _,pred= model(Xi,Xv)
            pred = list(pred)
            label = list(label)
            test_pred.extend(pred)
            true_label.extend(label)


        end = time.time()
        acc = accuracy_score(true_label,test_pred)
        if isinstance(acc,tuple):
            acc = acc[0]
        f1 = f1_score(true_label, test_pred, average='weighted')
        if f1 > max_f1:
            if save_name and max_acc > 0.80:
                torch.save(model.state_dict(), save_name)
            max_acc = acc
            max_f1 = f1
            max_epoch = epoch
        print(
            '[{}  {}] Train_Loss: {:.4f} Train_Acc: {:.4f} Test_Acc: {:.4f} Test_f1: {:.4f} Time: {:.2f}'.format(
            epoch + 1, epoches, train_loss, train_acc / count_label, acc, f1, (end - start)
            ),'\n',
            '{'+'Epoch: {} Max_acc: {:.4f} Max_F1: {:.4f}'.format(max_epoch+1, max_acc, max_f1)+'}'
        )


if __name__ == "__main__":

    # get_data_index('data/test_index.csv','test_index.csv')
    # get_data_index('data/train_index','train_index.csv')
    feat_sizes = [1, 3, 70, 11, 1, 1, 2, 1, 1, 1,
                  1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2,
                  2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1]

    train_index = 'data/train_index.csv'
    train_value = 'data/train_normal.csv'
    test_index = 'data/test_index.csv'
    test_value = 'data/test_normal.csv'
    model = dnn(41, feat_sizes, 8, 5, batch_size=400, lr=0.0055, optimer='adam',
                    epoches=5, weight_decay=8e-7)
    model.init_weight()
    model.fit(train_index, train_value,test_index,test_value,
              shuffle=True, print_params=True)
    # model.kflod_val(train_index,train_value,10,verbose=100)
    # model.load_state_dict(torch.load('adam_400_0055_max.pkl'))
    # model.predict(test_index,test_value)


