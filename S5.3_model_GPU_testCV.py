#v2.0.0
# _with_dataset2
from collections import defaultdict
import csv
import json
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
from torch_geometric.data import Data

from S4_dataloader import read_graph_data_from_file

# 设置随机种子以确保结果的可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 元素符号和原子序数的字典，所有元素都检查无误，之前元素周期表绘图同样来自此字典
element_numbers_dict = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
    'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
    'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
    'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
    'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
} 

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

# 获取csv中包含的所有阳离子元素
def get_elements_of_2dataset():
    ov_0_csvpath="./charge0_with_dataset2.csv"
    dft_data = pd.read_csv(ov_0_csvpath, index_col=None, header=0).values
    # 获取dft_data的第1列，即化合物名称 
    compound_list=dft_data[:,1]
    # 使用集合对compound_list进行去重
    compound_list_unique = list(set(compound_list))
    random.seed(72)
    random.shuffle(compound_list_unique)
    compounds = compound_list_unique

    # 创建一个字典来存储每个元素对应的化合物  {'Re': ['Na3ReO5', 'Mg(ReO4)2', 'CsReO4', 'Na2ReO3', 'RbReO4', 'ReAgO4', 'KReO4'],......}
    element_to_compounds = defaultdict(list)

    for compound in compounds:
        elements = extract_elements(compound)
        for element in elements:
            element_to_compounds[element].append(compound)

    elements_contianed=list(element_to_compounds.keys())   # 获取csv中包含的所有阳离子元素

    elements_in_periodicTable=list(element_numbers_dict.keys())
    elements_not_contianed=[]
    for element in elements_in_periodicTable:  # 获取csv中不包含的所有元素,O元素除外
        if (element not in elements_contianed) and (element!='O'):
            elements_not_contianed.append(element)

    elements_contianed= sorted(elements_contianed, key=lambda x: elements_in_periodicTable.index(x))
    elements_not_contianed= sorted(elements_not_contianed, key=lambda x: elements_in_periodicTable.index(x))
    return elements_contianed,elements_not_contianed

"""

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
        savepath=f"./result_GPU/split_by_vacancies/{i}/"
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)        
        train_loader,test_loader=read_graph_data_from_file(batch_size,x_csvpath_list,y_csvpath_list)
        model = GNNWithEdgeFeatures(num_node_features=32,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12  
        criterion = torch.nn.L1Loss()  # 假设是回归任务
        criterion_mae = torch.nn.L1Loss()  # 使用MAE作为损失函数 
        # 训练模型
        train_loss,test_loss=train(model, train_loader, test_loader, lr,weight_decay, criterion,criterion_mae,savepath)
        results.append(["split_by_vacancies",i,train_loss,test_loss])    #地址！！！
    df = pd.DataFrame(results, columns=["split_type", "fold", "train_loss", "test_loss"])   
    df.to_csv("./result_GPU/split_by_vacancies/allresult.csv", index=False)   #地址！！！
        
    # 按化合物划分 split_by_compounds--------------------------------------------------------------------------------------------

    for i in range(5):
        y_index=i
        y_csvpath_list = [f"./Kfold_split_with_dataset2/split_by_compounds/dft_data_subset_{i}.csv"]

        x_index = [j for j in range(5) if j != i]  # y_index是一个列表，包括了从0到4这5个整数除了x_index之外的所有整数
        x_csvpath_list = [f"./Kfold_split_with_dataset2/split_by_compounds/dft_data_subset_{j}.csv" for j in x_index]
        print("x_index=",x_index, x_csvpath_list)  
        print("y_index=",y_index, y_csvpath_list)
        savepath=f"./result_GPU/split_by_compounds/{i}/"
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)        
        train_loader,test_loader=read_graph_data_from_file(batch_size,x_csvpath_list,y_csvpath_list)
        model = GNNWithEdgeFeatures(num_node_features=32,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12  
        criterion = torch.nn.L1Loss()  # 假设是回归任务
        criterion_mae = torch.nn.L1Loss()  # 使用MAE作为损失函数 
        # 训练模型
        train_loss,test_loss=train(model, train_loader, test_loader, lr,weight_decay, criterion,criterion_mae,savepath)
        results.append(["split_by_compounds",i,train_loss,test_loss])    #地址！！！
    df = pd.DataFrame(results, columns=["split_type", "fold", "train_loss", "test_loss"])   
    df.to_csv("./result_GPU/split_by_compounds/allresult.csv", index=False)   #地址！！！
    
        
        

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
        savepath=f"./result_GPU/split_by_elements/{element}/"

        if not os.path.exists(savepath):
            os.makedirs(savepath)        
        train_loader,test_loader=read_graph_data_from_file(batch_size,x_csvpath_list,y_csvpath_list)
        model = GNNWithEdgeFeatures(num_node_features=32,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12  
        criterion = torch.nn.L1Loss()  # 假设是回归任务
        criterion_mae = torch.nn.L1Loss()  # 使用MAE作为损失函数 
        # 训练模型
        train_loss,test_loss=train(model, train_loader, test_loader, lr,weight_decay, criterion,criterion_mae,savepath)
        results.append(["split_by_elements",element,train_loss,test_loss])    #地址！！！
    df = pd.DataFrame(results, columns=["split_type", "fold", "train_loss", "test_loss"])   
    df.to_csv("./result_GPU/split_by_elements/allresult.csv", index=False)   #地址！！！    
    
    
    # 按元素数目划分 split_by_elements_numbers------------------------------------------------------------------------------------------------------------------
    for i in range(2):
        if i==0:
            x_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_1.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_2.csv"]
            y_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_3.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_4.csv"]
        else:
            x_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_1.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_3.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_2.csv"]
            y_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_4.csv"]
        savepath=f"./result_GPU/split_by_elements_numbers/{i}/"
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)        
        train_loader,test_loader=read_graph_data_from_file(batch_size,x_csvpath_list,y_csvpath_list)
        model = GNNWithEdgeFeatures(num_node_features=32,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12  
        criterion = torch.nn.L1Loss()  # 假设是回归任务
        criterion_mae = torch.nn.L1Loss()  # 使用MAE作为损失函数 
        # 训练模型
        train_loss,test_loss=train(model, train_loader, test_loader, lr,weight_decay, criterion,criterion_mae,savepath)
        results.append(["split_by_elements_numbers",i,train_loss,test_loss])    #地址！！！
    df = pd.DataFrame(results, columns=["split_type", "fold", "train_loss", "test_loss"])   
    df.to_csv("./result_GPU/split_by_elements_numbers/allresult.csv", index=False)   #地址！！！   
        
    #  最后汇总
    results_df = pd.DataFrame(results, columns=["split_type", "fold", "train_loss", "test_loss"])
    results_df.to_csv("./result_GPU/allresult.csv", index=False)
"""

# 按氧空位划分 split_by_vacancies-------------------------------------------------------------------------------------------
def load_and_test_cv_by_vacancies():
    # 调用函数设置随机种子
    set_seed(42) 
    # 定义超参数搜索空间
    heads = 4 
    hidden_dim = 64 # 隐藏特征单元
    batch_size =2 # 批处理大小
    lr =0.001  # 批处理大小0.0005
    dropout=0.3
    weight_decay=0.0001
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    csv_path=".\\result_GPU\\split_by_vacancies\\all_vacancies_result.csv"   #！！！！！ 注意地址！！！！！
    index=0
    # 打开CSV文件以追加模式（'a'）打开
    with open(csv_path, mode='a', newline='') as file:   #newline=''参数的作用是确保在不同操作系统上正确处理换行符。
        writer = csv.writer(file)
        writer.writerow(["index","model","crystal_name","Oxy_site_label","Oxy_site_index","Oxy_vacancy_formation_energy","Oxy_vacancy_formation_energy_pre"])        # 写入新的一行


    for i in range(5):
        # 载入模型>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        model = GNNWithEdgeFeatures(num_node_features=32,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12  
        model_folderpath=".\\result_GPU\\split_by_vacancies\\"+str(i)     #！！！！！ 注意地址！！！！！
        model_path=next(os.path.join(model_folderpath, f) for f in os.listdir(model_folderpath) if f.endswith("modeldict.pth"))
        print(model_path)
        model.load_state_dict(torch.load(model_path,weights_only=False,map_location=torch.device(device)))
        model.to(device)
        model.eval()  # 设置模型为评估模式

        # 设定数据路径>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        y_index=i
        y_csvpath_list = [f"./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_{i}.csv"]    #！！！！！ 注意地址！！！！！
                
        graph_data_path="./transformed_graph_with_dataset2/"
        dft_dataframe_y = pd.concat([pd.read_csv(file, index_col=None, header=0) for file in y_csvpath_list])

        # 遍历数据集并检测
        for vacancy_data in tqdm(dft_dataframe_y.values,colour="green",desc="Loading y data"):
            with open(graph_data_path+vacancy_data[2]+".json", 'r', encoding='utf-8') as file:
                this_vacancy_graphdata_list = json.load(file)
            for site_graph_data in this_vacancy_graphdata_list:
                """
                print("y, crystal_name:",site_graph_data["crystal_name"], 
                    "Oxy_site_label:",site_graph_data["Oxy_site_label"],
                    "Oxy_site_index:",site_graph_data["Oxy_site_index"],
                    "vacancy_formation_energy:",site_graph_data["Oxy_vacancy_formation_energy"])
                """
                y = torch.tensor(site_graph_data["Oxy_vacancy_formation_energy"], dtype=torch.float)
                x = torch.tensor(site_graph_data["node_features"], dtype=torch.float)
                source_nodes = [edge[0] for edge in site_graph_data["edges"]]
                target_nodes = [edge[1] for edge in site_graph_data["edges"]]
                edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
                edge_attr = torch.tensor(site_graph_data["edge_attr"], dtype=torch.float)
                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
                data.to(device)
                out = model(data)   # 前向传播
                out=out.cpu()  
                index+=1
                # 打开CSV文件以追加模式（'a'）打开
                with open(csv_path, mode='a', newline='') as file:   #newline=''参数的作用是确保在不同操作系统上正确处理换行符。
                    writer = csv.writer(file)
                    # ["index","model","crystal_name","Oxy_site_label","Oxy_site_index","Oxy_vacancy_formation_energy","Oxy_vacancy_formation_energy_pre"]
                    writer.writerow([index,model_path,site_graph_data["crystal_name"],site_graph_data["Oxy_site_label"],
                                     site_graph_data["Oxy_site_index"],site_graph_data["Oxy_vacancy_formation_energy"],out.item()])        # 写入新的一行
                            

# 按化合物划分 split_by_compounds-------------------------------------------------------------------------------------------
def load_and_test_cv_by_compounds():       #！！！！！ 注意名字！！！！！   
    # 调用函数设置随机种子
    set_seed(42) 
    # 定义超参数搜索空间
    heads = 4 
    hidden_dim = 64 # 隐藏特征单元
    batch_size =2 # 批处理大小
    lr =0.001  # 批处理大小0.0005
    dropout=0.3
    weight_decay=0.0001
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    csv_path=".\\result_GPU\\split_by_compounds\\all_vacancies_result.csv"   #！！！！！ 注意地址！！！！！
    index=0
    # 打开CSV文件以追加模式（'a'）打开
    with open(csv_path, mode='a', newline='') as file:   #newline=''参数的作用是确保在不同操作系统上正确处理换行符。
        writer = csv.writer(file)
        writer.writerow(["index","model","crystal_name","Oxy_site_label","Oxy_site_index","Oxy_vacancy_formation_energy","Oxy_vacancy_formation_energy_pre"])        # 写入新的一行


    for i in range(5):
        # 载入模型>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        model = GNNWithEdgeFeatures(num_node_features=32,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12  
        model_folderpath=".\\result_GPU\\split_by_compounds\\"+str(i)     #！！！！！ 注意地址！！！！！
        model_path=next(os.path.join(model_folderpath, f) for f in os.listdir(model_folderpath) if f.endswith("modeldict.pth"))
        print(model_path)
        model.load_state_dict(torch.load(model_path,weights_only=False,map_location=torch.device(device)))
        model.to(device)
        model.eval()  # 设置模型为评估模式

        # 设定数据路径>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        y_index=i
        y_csvpath_list = [f"./Kfold_split_with_dataset2/split_by_compounds/dft_data_subset_{i}.csv"]    #！！！！！ 注意地址！！！！！
                
        graph_data_path="./transformed_graph_with_dataset2/"
        dft_dataframe_y = pd.concat([pd.read_csv(file, index_col=None, header=0) for file in y_csvpath_list])

        # 遍历数据集并检测
        for vacancy_data in tqdm(dft_dataframe_y.values,colour="green",desc="Loading y data"):
            with open(graph_data_path+vacancy_data[2]+".json", 'r', encoding='utf-8') as file:
                this_vacancy_graphdata_list = json.load(file)
            for site_graph_data in this_vacancy_graphdata_list:
                """
                print("y, crystal_name:",site_graph_data["crystal_name"], 
                    "Oxy_site_label:",site_graph_data["Oxy_site_label"],
                    "Oxy_site_index:",site_graph_data["Oxy_site_index"],
                    "vacancy_formation_energy:",site_graph_data["Oxy_vacancy_formation_energy"])
                """
                y = torch.tensor(site_graph_data["Oxy_vacancy_formation_energy"], dtype=torch.float)
                x = torch.tensor(site_graph_data["node_features"], dtype=torch.float)
                source_nodes = [edge[0] for edge in site_graph_data["edges"]]
                target_nodes = [edge[1] for edge in site_graph_data["edges"]]
                edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
                edge_attr = torch.tensor(site_graph_data["edge_attr"], dtype=torch.float)
                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
                data.to(device)
                out = model(data)   # 前向传播
                out=out.cpu()  
                index+=1
                # 打开CSV文件以追加模式（'a'）打开
                with open(csv_path, mode='a', newline='') as file:   #newline=''参数的作用是确保在不同操作系统上正确处理换行符。
                    writer = csv.writer(file)
                    # ["index","model","crystal_name","Oxy_site_label","Oxy_site_index","Oxy_vacancy_formation_energy","Oxy_vacancy_formation_energy_pre"]
                    writer.writerow([index,model_path,site_graph_data["crystal_name"],site_graph_data["Oxy_site_label"],
                                     site_graph_data["Oxy_site_index"],site_graph_data["Oxy_vacancy_formation_energy"],out.item()])        # 写入新的一行
                    



# 按元素划分 split_by_elements-------------------------------------------------------------------------------------------
def load_and_test_cv_by_elements():       #！！！！！ 注意名字！！！！！   
    # 调用函数设置随机种子
    set_seed(42) 
    # 定义超参数搜索空间
    heads = 4 
    hidden_dim = 64 # 隐藏特征单元
    batch_size =2 # 批处理大小
    lr =0.001  # 批处理大小0.0005
    dropout=0.3
    weight_decay=0.0001
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elements_contianed,_= get_elements_of_2dataset()

    csv_path=".\\result_GPU\\split_by_elements\\all_vacancies_result.csv"   #！！！！！ 注意地址！！！！！
    index=0
    # 打开CSV文件以追加模式（'a'）打开
    with open(csv_path, mode='a', newline='') as file:   #newline=''参数的作用是确保在不同操作系统上正确处理换行符。
        writer = csv.writer(file)
        writer.writerow(["index","model","crystal_name","Oxy_site_label","Oxy_site_index","Oxy_vacancy_formation_energy","Oxy_vacancy_formation_energy_pre","test_elements"])        # 写入新的一行


    for element in elements_contianed:
        # 载入模型>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        model = GNNWithEdgeFeatures(num_node_features=32,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12  
        model_folderpath=".\\result_GPU\\split_by_elements\\"+str(element)     #！！！！！ 注意地址！！！！！
        model_path=next(os.path.join(model_folderpath, f) for f in os.listdir(model_folderpath) if f.endswith("modeldict.pth"))
        print(model_path)
        model.load_state_dict(torch.load(model_path,weights_only=False,map_location=torch.device(device)))
        model.to(device)
        model.eval()  # 设置模型为评估模式

        # 设定数据路径>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        y_index=element
        y_csvpath_list = [f"./Kfold_split_with_dataset2/split_by_elements/{y_index}_y.csv"]    #！！！！！ 注意地址！！！！！
                
        graph_data_path="./transformed_graph_with_dataset2/"
        dft_dataframe_y = pd.concat([pd.read_csv(file, index_col=None, header=0) for file in y_csvpath_list])

        # 遍历数据集并检测
        for vacancy_data in tqdm(dft_dataframe_y.values,colour="green",desc="Loading y data"):
            with open(graph_data_path+vacancy_data[2]+".json", 'r', encoding='utf-8') as file:
                this_vacancy_graphdata_list = json.load(file)
            for site_graph_data in this_vacancy_graphdata_list:
                """
                print("y, crystal_name:",site_graph_data["crystal_name"], 
                    "Oxy_site_label:",site_graph_data["Oxy_site_label"],
                    "Oxy_site_index:",site_graph_data["Oxy_site_index"],
                    "vacancy_formation_energy:",site_graph_data["Oxy_vacancy_formation_energy"])
                """
                y = torch.tensor(site_graph_data["Oxy_vacancy_formation_energy"], dtype=torch.float)
                x = torch.tensor(site_graph_data["node_features"], dtype=torch.float)
                source_nodes = [edge[0] for edge in site_graph_data["edges"]]
                target_nodes = [edge[1] for edge in site_graph_data["edges"]]
                edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
                edge_attr = torch.tensor(site_graph_data["edge_attr"], dtype=torch.float)
                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
                data.to(device)
                out = model(data)   # 前向传播
                out=out.cpu()  
                index+=1
                # 打开CSV文件以追加模式（'a'）打开
                with open(csv_path, mode='a', newline='') as file:   #newline=''参数的作用是确保在不同操作系统上正确处理换行符。
                    writer = csv.writer(file)
                    # ["index","model","crystal_name","Oxy_site_label","Oxy_site_index","Oxy_vacancy_formation_energy","Oxy_vacancy_formation_energy_pre"]  # 注意按元素的要多加一个
                    writer.writerow([index,model_path,site_graph_data["crystal_name"],site_graph_data["Oxy_site_label"],
                                     site_graph_data["Oxy_site_index"],site_graph_data["Oxy_vacancy_formation_energy"],out.item(),y_index])        # 写入新的一行
    



# 按元素数目划分 split_by_elements_numbers-------------------------------------------------------------------------------------------
def load_and_test_cv_by_elements_numbers():       #！！！！！ 注意名字！！！！！   
    # 调用函数设置随机种子
    set_seed(42) 
    # 定义超参数搜索空间
    heads = 4 
    hidden_dim = 64 # 隐藏特征单元
    batch_size =2 # 批处理大小
    lr =0.001  # 批处理大小0.0005
    dropout=0.3
    weight_decay=0.0001
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elements_contianed,_= get_elements_of_2dataset()

    csv_path=".\\result_GPU\\split_by_elements_numbers\\all_vacancies_result.csv"   #！！！！！ 注意地址！！！！！
    index=0
    # 打开CSV文件以追加模式（'a'）打开
    with open(csv_path, mode='a', newline='') as file:   #newline=''参数的作用是确保在不同操作系统上正确处理换行符。
        writer = csv.writer(file)
        writer.writerow(["index","model","crystal_name","Oxy_site_label","Oxy_site_index","Oxy_vacancy_formation_energy","Oxy_vacancy_formation_energy_pre","test_elements_numbers"])        # 写入新的一行


    for i in range(1,4):
        # 载入模型>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        model = GNNWithEdgeFeatures(num_node_features=32,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为12  
        model_folderpath=".\\result_GPU\\split_by_elements_numbers\\"+str(i)     #！！！！！ 注意地址！！！！！
        model_path=next(os.path.join(model_folderpath, f) for f in os.listdir(model_folderpath) if f.endswith("modeldict.pth"))
        print(model_path)
        model.load_state_dict(torch.load(model_path,weights_only=False,map_location=torch.device(device)))
        model.to(device)
        model.eval()  # 设置模型为评估模式

        # 设定数据路径>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    
        if i==1:                                                                                      #！！！！！ 注意地址！！！！！
            y_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_2.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_3.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_4.csv"]
        elif i==2:
            y_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_3.csv",
                              "./Kfold_split_with_dataset2/split_by_elements_numbers/eles_4.csv"]
        else:
            y_csvpath_list = ["./Kfold_split_with_dataset2/split_by_elements_numbers/eles_4.csv"]
            
        graph_data_path="./transformed_graph_with_dataset2/"
        dft_dataframe_y = pd.concat([pd.read_csv(file, index_col=None, header=0) for file in y_csvpath_list])

        # 遍历数据集并检测
        for vacancy_data in tqdm(dft_dataframe_y.values,colour="green",desc="Loading y data"):
            with open(graph_data_path+vacancy_data[2]+".json", 'r', encoding='utf-8') as file:
                this_vacancy_graphdata_list = json.load(file)
            for site_graph_data in this_vacancy_graphdata_list:
                """
                print("y, crystal_name:",site_graph_data["crystal_name"], 
                    "Oxy_site_label:",site_graph_data["Oxy_site_label"],
                    "Oxy_site_index:",site_graph_data["Oxy_site_index"],
                    "vacancy_formation_energy:",site_graph_data["Oxy_vacancy_formation_energy"])
                """
                y = torch.tensor(site_graph_data["Oxy_vacancy_formation_energy"], dtype=torch.float)
                x = torch.tensor(site_graph_data["node_features"], dtype=torch.float)
                source_nodes = [edge[0] for edge in site_graph_data["edges"]]
                target_nodes = [edge[1] for edge in site_graph_data["edges"]]
                edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
                edge_attr = torch.tensor(site_graph_data["edge_attr"], dtype=torch.float)
                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
                data.to(device)
                out = model(data)   # 前向传播
                out=out.cpu()  
                index+=1
                # 打开CSV文件以追加模式（'a'）打开
                with open(csv_path, mode='a', newline='') as file:   #newline=''参数的作用是确保在不同操作系统上正确处理换行符。
                    writer = csv.writer(file)
                    # ["index","model","crystal_name","Oxy_site_label","Oxy_site_index","Oxy_vacancy_formation_energy","Oxy_vacancy_formation_energy_pre"]  # 注意按元素的要多加一个
                    writer.writerow([index,model_path,site_graph_data["crystal_name"],site_graph_data["Oxy_site_label"],
                                     site_graph_data["Oxy_site_index"],site_graph_data["Oxy_vacancy_formation_energy"],out.item(),
                                     len(extract_elements(site_graph_data["crystal_name"]))])        # 写入新的一行
    


    
if __name__ == "__main__":
    #train_allsplit()
    #load_and_test_cv_by_vacancies()
    #load_and_test_cv_by_compounds()
    #load_and_test_cv_by_elements()
    load_and_test_cv_by_elements_numbers()

    #print(extract_elements("Al2CuO3"))