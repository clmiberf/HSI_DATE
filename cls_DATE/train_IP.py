from email.mime import image
from telnetlib import X3PAD
from tkinter import E
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
#import spectral
import time
import spectral
import imageio
from regionvit import ViT
import torch.backends.cudnn as cudnn
#from vit_pytorch import ViT

PATCH_SIZE = 15
PCA_NUM = 50
NUM_CLASS = 16
DATASETS = "IP"

# Parameter Setting
seed =3
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cudnn.deterministic = True
cudnn.benchmark = False


#20 37 20 14 27 39 36 113 62 33 11 19 9 11 72 18
def loadData():
    # 读入数据
    if(DATASETS == "IP"):
        data = sio.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']
    elif(DATASETS == "PU"):
        data = sio.loadmat('../data/PaviaU.mat')['paviaU']
        labels = sio.loadmat('../data/PaviaU_gt.mat')['paviaU_gt']
    elif(DATASETS == "SA"):
        data = sio.loadmat('../data/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('../data/Salinas_gt.mat')['salinas_gt']

    return data, labels

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1

    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels




def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

BATCH_SIZE_TRAIN = 64

def create_data_loader():
    # 地物类别
    # class_num = 16
    # 读入数据
    X, labels = loadData()
    y = labels
    print(y)
    # 用于测试样本的比例
    if(DATASETS == "IP"):
        test_ratio = 0.90
    elif(DATASETS == "PU"):
        test_ratio = 0.97
    elif(DATASETS == "SA"):
        test_ratio = 0.99
  
    # 每个像素周围提取 patch 的尺寸
    patch_size = PATCH_SIZE
    # 使用 PCA 降维，得到主成分的数量
    pca_components = PCA_NUM

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    input = X
    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:,:,i])
        input_min = np.min(input[:,:,i])
        input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
    X = input_normalize

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    

    print('\n... ... create data cubes ... ...')
    X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
    Xvaild,_,Yvaild,_ = splitTrainTestSet(Xtest, ytest, 0.80)
    print('Xtrain shape: ', Xtrain.shape)
    print('Ytrain shape: ',ytrain.shape)
    print('Xvaild shape: ',Xvaild.shape)
    for i in range(int(np.max(ytrain))+1):

        print(np.sum(ytrain==i))
    print(np.max(ytrain))
    print('Xtest  shape: ', Xtest.shape)
    for i in range(int(np.max(ytrain))+1):

       print(np.sum(ytest==i))

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components)
    Xvaild = Xvaild.reshape(-1, patch_size, patch_size, pca_components)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0, 3, 1, 2)
#    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
#    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 3, 1, 2)
    Xtest = Xtest.transpose(0, 3, 1, 2)
    Xvaild = Xvaild.transpose(0, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)
    print('after transpose: Xvaild  shape: ', Xvaild.shape)

    # 创建train_loader和 test_loader
    Y = X
    X = TestDS(X, y)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    vaildset = TestDS(Xvaild, Yvaild)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               drop_last=True
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=True
                                              )
    vaild_loader = torch.utils.data.DataLoader(dataset=vaildset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=True
                                              )                                           
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=True
                                              )

    return train_loader, test_loader,vaild_loader , all_data_loader, labels,Y

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len



def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    if(DATASETS == "IP"):
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']
    elif(DATASETS == "PU"):
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees'
        , 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                    'Self-Blocking Bricks', 'Shadows']
    elif(DATASETS == "SA"):
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow'
        , 'Fallow_smooth', 'Stubble', 'Celery',
                    'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
                    'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 'Vinyard_untrained',
                    'Vinyard_vertical_trellis']


    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100


def train(train_loader,test_loader,dropout,emb_dropout,epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
#    net = SSFTTnet.SSFTTnet().cuda()

    net = ViT(
        image_size = PATCH_SIZE,
        near_band =1,
        num_regions = PCA_NUM,
        num_patches=7,
        num_classes = NUM_CLASS,
        dim = 64,
        depth = 7,
        heads = 4,
        mlp_dim = 4,
        dropout = dropout,
        emb_dropout = emb_dropout,
    ).cuda()
    
    # net = ViT(
    #     image_size = PATCH_SIZE,
    #     near_band = 1,
    #     num_patches = PCA_NUM,
    #     num_classes = NUM_CLASS,
    #     dim = 64,
    #     depth = 7,
    #     heads = 4,
    #     mlp_dim = 4,
    #     dropout = 0.1,
    #     emb_dropout = 0.3,
    #     mode = 'CAF'
    # ).cuda()

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0
    best_oa = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
          #  print(data.shape)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data)
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
        
                                                                         loss.item()))
        y_pred_test, y_test = test(device, net, test_loader)
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
        print('OA: ',oa)
        print('AA: ',aa)
        print('K: ',kappa)
        print('best aa: ',best_oa)
        if oa > best_oa:
            best_oa = oa
            torch.save(net.state_dict(),'SSFTTnet_params_ip.pth')

    print('Finished Training')

    return net, device

def visionable_map_2(label_pred,labels):
    H = labels.shape[0]
    W = labels.shape[1]
    print(H)
    print(W)
    print(labels.shape)
    print(labels)
    prediction = np.zeros((H,W))
  
    #index = 0
    ind = 0
    for r in range(H):
        for c in range(W):
            target = labels[r,c]
            #print(target)
            if(target == 0):
                #y[index] = np.array([250, 250, 250]) / 255.
                continue
            elif ind < 10240:
                prediction[r][c] = label_pred[ind]+1
                
                ind += 1
  
    print(prediction)
    spectral.save_rgb("predictions_dat_ip.png", prediction.astype(int), colors=spectral.spy_colors)

    return prediction


def classfication_map(net,X,labels,patch_size):
     # 给 X 做 padding
#    margin = int((patch_size - 1) / 2)
    
    print(labels)
    H = labels.shape[0]
    W = labels.shape[1]
    prediction = np.zeros((H,W))
    patchIndex = 0
    index = 0
    for r in range(H):
        for c in range(W):
            target = labels[r,c]
            if(target == 0):
                continue
            else:
                input = X[index,:,:,:]
                #print(X.shape)
                input = torch.Tensor(input).unsqueeze(0)
                input = input.cuda()
                outputs = net(input)
                outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                prediction[r][c] = outputs+2
               # print(prediction[r][c])
               # print(target)
               # print("******")
                index += 1
    #predict_image = spectral.imshow(classes = prediction.astype(int),figsize =(7,7))
    spectral.save_rgb("predictions_ip.png", prediction.astype(int), colors=spectral.spy_colors)
    return prediction


if __name__ == '__main__':

    if(DATASETS == "IP"):
        dropout=0.1
        emb_dropout=0.3
        params_path = 'SSFTTnet_params_ip.pth'
        file_name = "cls_result/classification_report_region_ip.txt"
    elif(DATASETS == "PU"):
        dropout=0.1
        emb_dropout=0.2
        params_path = 'SSFTTnet_params_pu.pth'
        file_name = "cls_result/classification_report_region_pu_2.txt"
    elif(DATASETS == "SA"):
        dropout=0.1
        emb_dropout=0.1
        params_path = 'SSFTTnet_params_sa.pth'
        file_name = "cls_result/classification_report_region_sa_2.txt"

    train_loader, test_loader, vaild_loader,all_data_loader, y_all,X= create_data_loader()
    tic1 = time.perf_counter()
    net, device = train(train_loader,test_loader,dropout=dropout,emb_dropout=emb_dropout,epochs=70)
    # 只保存模型参数
    
    #torch.save(net.state_dict(),'SSFTTnet_params_pu.pth')
    net.load_state_dict(torch.load('./SSFTTnet_params_ip.pth')) 
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    # 评价指标
    
#    net.load_state_dict(torch.load('./SSFTTnet_params_ip.pth')) 
#    y_pred_test, y_test = test(device, net, test_loader)
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
  #  file_name = "cls_result/classification_report_region_sa.txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

#    net.load_state_dict(torch.load('./SSFTTnet_params_pu.pth'))  
    print("##################################")
    #classfication_map(net=net,X=X,labels=y_all,patch_size=15)
    y_pred_test, y_test = test(device, net, all_data_loader)
#    print("lllllll")
    print(y_test.shape)
#  image = ColorResult(y_pred_test,y_all)
    image = visionable_map_2(y_pred_test,y_all)



   

 



