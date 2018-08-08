import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


num_feats = ['duration','src_bytes','dst_bytes','wrong_fragment',
            'urgent','hot','num_failed_logins','num_compromised','su_attempted','num_root',
           'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
           'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
            'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
            'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
           'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',]

index_feats = ['protocol_type','service','flag','land','logged_in','root_shell','is_host_login','is_guest_login',]

def normal_data(train,test):
    std = StandardScaler()
    for feat in num_feats:
        train[feat] = std.fit_transform(train[feat].reshape((-1,1)))
        test[feat] = std.transform(test[feat].reshape((-1,1)))

    return train,test

def label_data(train,test):
    train['num'] =0
    test['num'] = 1
    data = pd.concat([train,test],axis=0)
    for feat in index_feats:
        enc = LabelEncoder()
        data[feat] = enc.fit_transform(data[feat])

    train = data[data['num'] == 0]
    test = data[data['num'] == 1]

    return train,test

def scaler_data(train,test):
    mms = MinMaxScaler(feature_range=(0,1))
    for feat in num_feats:
        train[feat] = mms.fit_transform(train[feat].reshape((-1,1)))
        test[feat] = mms.transform(test[feat].reshape((-1,1)))

    return train, test

def make_index(train,test):
    train['num'] = 0
    test['num'] = 1
    data = pd.concat([train, test], axis=0)
    for feat in num_feats:
        data[feat] = 0

    train = data[data['num'] == 0]
    test = data[data['num'] == 1]

    return train, test

def trans_data(train,test,method=None):
    train,test = label_data(train,test)
    if method:
        if method == 'normal':
            train,test = normal_data(train,test)
        elif method == 'scaler':
            train,test = scaler_data(train,test)
        elif method == 'raw':
            train, test = train, test
        else:
            print('method error')
            return None

    train.drop('num', axis=1,inplace=True)
    test.drop('num', axis=1,inplace=True)
    train.to_csv('data/'+'train_'+str(method)+'.csv',index=False,float_format='%.8f')
    test.to_csv('data/'+'test_' + str(method) + '.csv', index=False,float_format='%.8f')

    train_index, test_index = make_index(train, test)
    train_index.drop(['label','num'], axis=1, inplace=True)
    test_index.drop(['label','num'], axis=1, inplace=True)
    train_index.to_csv('data/train_index.csv',index=False)
    test_index.to_csv('data/test_index.csv',index=False)




if __name__ == "__main__":
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    trans_data(train_data,test_data,'scaler')
    # data = pd.concat([train_data,test_data],axis=0)
    # for i in data.columns:
    #     if i in num_feats:
    #         print(data[i].max())
    # feat_size = []
    # for i in data.columns:
    #     if i in index_feats:
    #         feat_size.append(data[i].max()+1)
    #     else:
    #         feat_size.append(1)
    # print(feat_size)
    # print(len(feat_size))
