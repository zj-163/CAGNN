#v2.0.0
# _with_dataset2
import os
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv,GCNConv, GATConv, global_mean_pool, LayerNorm
from torch.nn import Linear, Sequential, ReLU
from tqdm import tqdm

from S8_2_compare_dataloader_onlyStructure import read_graph_data_from_file

# 设置随机种子以确保结果的可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GNNWithEdgeFeatures(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim,heads,dropout,nnconv_aggr):
        super(GNNWithEdgeFeatures, self).__init__()
        self.dropout=dropout
        
        # 定义边网络
        self.edge_net = Sequential(Linear(num_edge_features, hidden_dim * num_node_features),
                                   ReLU(),
                                   Linear(hidden_dim * num_node_features, hidden_dim * num_node_features))
        self.conv1 = NNConv(num_node_features, hidden_dim, self.edge_net, aggr=nnconv_aggr)
        self.norm1 = LayerNorm(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads,edge_dim=2, concat=True)  # 注意力机制层
        self.norm2 = LayerNorm(hidden_dim)  # 因为GAT concat=True，输出维度是 hidden_dim
        self.lin = Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # NNConv层 + 激活函数 + 层次归一化
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT层 + 激活函数 + 层次归一化
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.norm2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 全局池化层
        x = global_mean_pool(x, batch)

        # 全连接层输出
        x = self.lin(x)

        return x




def train(model, train_loader, test_loader,lr,weight_decay, criterion,criterion_mae,savepath):
    # 检查CUDA是否可用，将模型移动到指定的设备，GPU或CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=0.001, threshold_mode="abs", min_lr=0.0001,patience=5, verbose=True)

    test_losses_mae=[]
    train_losses=[]
    learing_rates=[]
    # 训练模型
    for epoch in tqdm(range(150), desc="Epoch"):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()  # 清除之前的梯度
            data = data.to(device)
            out = model(data)   # 前向传播
            loss = criterion(out, data.y.view(-1, 1))  # 计算损失
            loss.backward()     # 反向传播，计算当前梯度
            optimizer.step()    # 根据梯度更新模型参数
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        #print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}")

        # 训练完一个epoch后，如果当前loss小于之前的最小loss，则在测试集上测试，并保存当前模型
        if (train_losses and  train_loss<min(train_losses)) or  epoch%10==9:
            model.eval()
            total_loss = 0
            total_loss_mae=0
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data)
                    loss = criterion(out, data.y.view(-1, 1))
                    loss_mae=criterion_mae(out, data.y.view(-1, 1))
                    total_loss += loss.item()
                    total_loss_mae+=loss_mae.item()
            test_loss = total_loss / len(test_loader)
            test_loss_mae=total_loss_mae/len(test_loader)
            test_losses_mae.append(test_loss_mae)
            tqdm.write(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f},  Val Loss_mae: {test_loss_mae:.4f},Learning Rate = {scheduler.get_last_lr()[0]}")
            #torch.save(model, './best_model.pth')
        else:
            test_losses_mae.append("None")

        train_losses.append(train_loss)
        learing_rates.append(scheduler.get_last_lr()[0])
        
        #scheduler.step()  # 更新学习率
        scheduler.step(train_loss)
        
        
    # 保存模型
    torch.save(model.state_dict(), savepath+ 'trainloss'+str(train_loss)+'_testmaeloss'+str(test_losses_mae[-1])+'_modeldict.pth')
    torch.save(model,savepath+ 'trainloss'+str(train_loss)+'_testmaeloss'+str(test_losses_mae[-1])+'_model.pth')
    # 将train_losses和test_losses_mae保存为csv文件，并加上表头和序号
    df=pd.DataFrame({"train_loss":train_losses,"test_loss_mae":test_losses_mae,"learning_rate":learing_rates})
    df.to_csv(savepath+"model_train.csv",index=True)
    return train_loss,test_losses_mae[-1]





# 解析化合物中的元素（不包括氧元素）、元素数量（未设置返回）
def extract_elements(compound):
    elements = set()
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', compound)    # ([A-Z][a-z]*)：匹配一个元素符号，(\d*)：匹配元素数量
    for match in matches:
        element = match[0]
        count = match[1]
        if count == '':
            count = '1'
        elements.add(element)
    elements.discard('O') # 删除元素O
    return list(elements)

def train_allsplit():
    # 调用函数设置随机种子
    set_seed(42) 
    # 定义超参数搜索空间
    heads = 4 
    hidden_dim = 64 # 隐藏特征单元
    batch_size =2 # 批处理大小
    lr =0.001  # 批处理大小0.0005
    dropout=0.3
    weight_decay=0.0001
    
    results=[]
    
    
    # 按氧空位划分 split_by_vacancies-------------------------------------------------------------------------------------------
    
    for i in range(5):
        y_index=i
        y_csvpath_list = [f"./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_{i}.csv"]    #！！！！！ 注意地址

        x_index = [j for j in range(5) if j != i]  # y_index是一个列表，包括了从0到4这5个整数除了x_index之外的所有整数
        x_csvpath_list = [f"./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_{j}.csv" for j in x_index]     #！！！！！ 注意地址
        print("x_index=",x_index, x_csvpath_list)  
        print("y_index=",y_index, y_csvpath_list)
        savepath=f"./S8_compare_data/ovgnn_only_structure/split_by_vacancies/{i}/"
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)        
        train_loader,test_loader=read_graph_data_from_file(batch_size,x_csvpath_list,y_csvpath_list)
        model = GNNWithEdgeFeatures(num_node_features=1,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12  
        criterion = torch.nn.L1Loss()  # 假设是回归任务
        criterion_mae = torch.nn.L1Loss()  # 使用MAE作为损失函数 
        # 训练模型
        train_loss,test_loss=train(model, train_loader, test_loader, lr,weight_decay, criterion,criterion_mae,savepath)
        results.append(["split_by_vacancies",i,train_loss,test_loss])    #地址！！！
    df = pd.DataFrame(results, columns=["split_type", "fold", "train_loss", "test_loss"])   
    df.to_csv("./S8_compare_data/ovgnn_only_structure/split_by_vacancies/allresult.csv", index=False)   #地址！！！
        
    # 按化合物划分 split_by_compounds--------------------------------------------------------------------------------------------

    for i in range(5):
        y_index=i
        y_csvpath_list = [f"./Kfold_split_with_dataset2/split_by_compounds/dft_data_subset_{i}.csv"]

        x_index = [j for j in range(5) if j != i]  # y_index是一个列表，包括了从0到4这5个整数除了x_index之外的所有整数
        x_csvpath_list = [f"./Kfold_split_with_dataset2/split_by_compounds/dft_data_subset_{j}.csv" for j in x_index]
        print("x_index=",x_index, x_csvpath_list)  
        print("y_index=",y_index, y_csvpath_list)
        savepath=f"./S8_compare_data/ovgnn_only_structure/split_by_compounds/{i}/"
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)        
        train_loader,test_loader=read_graph_data_from_file(batch_size,x_csvpath_list,y_csvpath_list)
        model = GNNWithEdgeFeatures(num_node_features=1,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12  
        criterion = torch.nn.L1Loss()  # 假设是回归任务
        criterion_mae = torch.nn.L1Loss()  # 使用MAE作为损失函数 
        # 训练模型
        train_loss,test_loss=train(model, train_loader, test_loader, lr,weight_decay, criterion,criterion_mae,savepath)
        results.append(["split_by_compounds",i,train_loss,test_loss])    #地址！！！
    df = pd.DataFrame(results, columns=["split_type", "fold", "train_loss", "test_loss"])   
    df.to_csv("./S8_compare_data/ovgnn_only_structure/split_by_compounds/allresult.csv", index=False)   #地址！！！
    
        
        

    # 按元素划分 split_by_elements------------------------------------------------------------------------------------------------------------------
    ov_0_csvpath="./charge0_with_dataset2.csv"
    dft_data = pd.read_csv(ov_0_csvpath, index_col=None, header=0).values
    # 获取dft_data的第1列，即化合物名称 
    compound_list=dft_data[:,1]
    # 使用集合对compound_list进行去重
    compound_list_unique = list(set(compound_list))
    random.shuffle(compound_list_unique)
    compounds = compound_list_unique

    # 创建一个字典来存储每个元素对应的化合物
    all_elements = []

    for compound in compounds:
        elements = extract_elements(compound)
        for element in elements:
            all_elements.append(element)
    all_elements=list(set(all_elements))
    print(all_elements)
    
    for element in all_elements:
        x_csvpath_list = [f"./Kfold_split_with_dataset2/split_by_elements/{element}_x.csv"]
        y_csvpath_list = [f"./Kfold_split_with_dataset2/split_by_elements/{element}_y.csv"]
        print("x_csvpath_list=",x_csvpath_list)  
        print("x_csvpath_list=",y_csvpath_list)
        savepath=f"./S8_compare_data/ovgnn_only_structure/split_by_elements/{element}/"

        if not os.path.exists(savepath):
            os.makedirs(savepath)        
        train_loader,test_loader=read_graph_data_from_file(batch_size,x_csvpath_list,y_csvpath_list)
        model = GNNWithEdgeFeatures(num_node_features=1,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12  
        criterion = torch.nn.L1Loss()  # 假设是回归任务
        criterion_mae = torch.nn.L1Loss()  # 使用MAE作为损失函数 
        # 训练模型
        train_loss,test_loss=train(model, train_loader, test_loader, lr,weight_decay, criterion,criterion_mae,savepath)
        results.append(["split_by_elements",element,train_loss,test_loss])    #地址！！！
    df = pd.DataFrame(results, columns=["split_type", "fold", "train_loss", "test_loss"])   
    df.to_csv("./S8_compare_data/ovgnn_only_structure/split_by_elements/allresult.csv", index=False)   #地址！！！    
    """
    
    # 按元素数目划分 split_by_elements_numbers------------------------------------------------------------------------------------------------------------------
    for i in range(1,4):
        if i==1:
            x_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_1.csv"
                              ]
            y_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_3.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_4.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_2.csv"]
        elif i==2:
            x_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_1.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_2.csv"]
            y_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_3.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_4.csv"]
        else:
            x_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_1.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_2.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_3.csv"]
            y_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_4.csv"]
        savepath=f"./S8_compare_data/ovgnn_only_structure/split_by_elements_numbers/{i}/"
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)        
        train_loader,test_loader=read_graph_data_from_file(batch_size,x_csvpath_list,y_csvpath_list)
        model = GNNWithEdgeFeatures(num_node_features=1,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12  
        criterion = torch.nn.L1Loss()  # 假设是回归任务
        criterion_mae = torch.nn.L1Loss()  # 使用MAE作为损失函数 
        # 训练模型
        train_loss,test_loss=train(model, train_loader, test_loader, lr,weight_decay, criterion,criterion_mae,savepath)
        results.append(["split_by_elements_numbers",i,train_loss,test_loss])    #地址！！！
    df = pd.DataFrame(results, columns=["split_type", "fold", "train_loss", "test_loss"])   
    df.to_csv("./S8_compare_data/ovgnn_only_structure/split_by_elements_numbers/allresult.csv", index=False)   #地址！！！   
        
    #  最后汇总
    results_df = pd.DataFrame(results, columns=["split_type", "fold", "train_loss", "test_loss"])
    #results_df.to_csv("./S8_compare_data/ovgnn_only_structure/allresult.csv", index=False)
    """
    
def train_with_alldata() : # 注意这里的y_csvpath_list仅用于凑train的输入参数，无任何实际检验意义
    # 调用函数设置随机种子
    set_seed(42) 
    # 定义超参数搜索空间
    heads = 4 
    hidden_dim = 64 # 隐藏特征单元
    batch_size =2 # 批处理大小
    lr =0.001  # 批处理大小0.0005
    dropout=0.3
    weight_decay=0.0001
    
    x_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_1.csv",
                        "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_2.csv",
                        "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_3.csv",
                        "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_4.csv"]
    y_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_4.csv"]
    
    savepath="./S8_compare_data/ovgnn_only_structure/train_with_alldata/"
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)        
    train_loader,test_loader=read_graph_data_from_file(batch_size,x_csvpath_list,y_csvpath_list)
    model = GNNWithEdgeFeatures(num_node_features=1,num_edge_features=2,                                
                            hidden_dim=hidden_dim,heads=heads,
                            dropout=dropout,
                            nnconv_aggr="add")  # 节点特征维度为32
    criterion = torch.nn.L1Loss()  # 假设是回归任务
    criterion_mae = torch.nn.L1Loss()  # 使用MAE作为损失函数 
    # 训练模型
    train_loss,test_loss=train(model, train_loader, test_loader, lr,weight_decay, criterion,criterion_mae,savepath)
        
    
    
    
if __name__ == "__main__":
    train_allsplit()
    #train_with_alldata()
    
    
    """
    # 调用函数设置随机种子
    set_seed(42) 

    # 定义超参数搜索空间
    heads = 4 
    hidden_dim = 64 # 隐藏特征单元
    batch_size =2 # 批处理大小
    lr =0.001  # 批处理大小0.0005
    dropout=0.3
    weight_decay=0.0001
    ov_0_x_csvpath_list=["./Kfold_split/split_by_vacancies/dft_data_subset_0.csv",
                         "./Kfold_split/split_by_vacancies/dft_data_subset_1.csv",
                         "./Kfold_split/split_by_vacancies/dft_data_subset_3.csv",
                         "./Kfold_split/split_by_vacancies/dft_data_subset_4.csv"]
    ov_0_y_csvpath_list=["./Kfold_split/split_by_vacancies/dft_data_subset_2.csv"]
    train_loader,test_loader=read_graph_data_from_file(batch_size,ov_0_x_csvpath_list,ov_0_y_csvpath_list)

    # 实例化模型和优化器
    model = GNNWithEdgeFeatures(num_node_features=1,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # 每50个epoch学习率乘以0.5
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,threshold=0.01,threshold_mode="abs", patience=5, verbose=True)
    criterion = torch.nn.L1Loss()  # 假设是回归任务
    #criterion = torch.nn.SmoothL1Loss()  # 假设是回归任务
    criterion_mae = torch.nn.L1Loss()  # 使用MAE作为损失函数
    savepath="./"
    # 训练模型
    val_mse=train(model, train_loader, test_loader, lr,weight_decay, criterion,criterion_mae,savepath)
    """

    

    