# 用于临时预测几个晶体，输入和结果都在/test/temptest文件夹下

import glob
import json
import pandas as pd
import numpy as np
import os
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.analysis.local_env import NearNeighbors,BrunnerNN_real,IsayevNN,ValenceIonicRadiusEvaluator
from multiprocessing import Pool
import warnings
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
import S5_model_GPU
from torch_geometric.data import Data


def structure2graph(this_structure,this_O_label,this_name,crystal_index,y):
    """
    注意，这里的代码要完全从S3中复制过来，不可以修改
    """
    results=[]
    for i_atom_0 in range(len(this_structure)):
        if this_structure[i_atom_0].label==this_O_label:    # 判断是否为一个O空位，如果是则返回一个结构，如果不是则不进行任何操作
            node_features=[]    #[0元素序号,2周期,3族,4是否镧系,5是否锕系,6电负性,7范德华半径,9原子半径-传统方法测量值]
            edges =[]   # 对于无向图，同一条边要表示两次，如（2，5）和（5，2）,但是不需要手动重复，因为每个原子都要遍历一遍
            edge_attr=[]
            df_elements_props = pd.read_csv('./Elements_props_all_v1.csv', header=0)
            for i_atom_1 in range(len(this_structure)):
                # 1. 构建节点特征矩阵-------------------------------------------
                this_feature=df_elements_props.iloc[this_structure[i_atom_1].specie.Z-1,[0,2,3,4,5,6,7,9,15,16,19,21]].values

                if i_atom_0==i_atom_1:    # 判断是否为特定O空位
                    this_feature=np.append(this_feature,1)  
                else:
                    this_feature=np.append(this_feature,0)  

                electron_orbital_energy=json.loads(df_elements_props.iloc[this_structure[i_atom_1].specie.Z-1,14].replace("'", '"')  )#将字符串中的单引号替换为双引号，严格遵循JSON格式
                
                electron_shelling_order=["1s","2s","2p","3s","3p","4s","3d","4p","5s","4d","5p","6s","4f","5d","6p","7s","5f","6d","7p"]
                for orbital in electron_shelling_order:
                    if orbital in electron_orbital_energy.keys():
                        this_feature=np.append(this_feature,electron_orbital_energy[orbital])
                    else:
                        this_feature=np.append(this_feature,0)
                #print(this_feature)

                if  np.isnan(this_feature.astype(float)).any() != False: # 检查是否有缺失值
                    print(this_feature.astype(float))
                    raise ValueError('存在缺失值')
                node_features.append (this_feature.astype(float).tolist())



                # 2. 获取邻接信息构建连接矩阵和边特征矩阵------------------------
                # 初始化一个最近邻分析类，
                nn = IsayevNN ()    #判断键合关系方法来源：https://www.nature.com/articles/ncomms15679
                # 获取i_atom_1原子的最近邻信息
                all_neighbors = nn.get_nn_info(this_structure,i_atom_1)
                
                #构建连接矩阵和边特征矩阵
                for neighbor in all_neighbors:
                    #print(neighbor)
                    edges.append((i_atom_1,int(neighbor["site_index"])))
                    if i_atom_1==i_atom_0 or neighbor["site_index"]==i_atom_0:
                        edge_attr.append([float(this_structure.get_distance(i_atom_1,neighbor["site_index"])),1])
                    else:
                        edge_attr.append([float(this_structure.get_distance(i_atom_1,neighbor["site_index"])),0])
            results.append({"crystal_index":crystal_index,"crystal_name":this_name,
                            "Oxy_site_label":this_O_label,"Oxy_site_index":i_atom_0,"Oxy_vacancy_formation_energy":y,
                            "node_features":node_features,"edges":edges,
                            "edge_attr":edge_attr})
    return results



def trans_one_cif(cif_path,cif_name):
    #print(cif_name)
    graph_results=[]
    struct= Structure.from_file(cif_path)
    # 遍历 Structure 中的所有原子
    oxygen_labels = []    # ['O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15', 'O16', 'O17', 'O18', 'O19', 'O20', 'O21']
    #print(struct)
    for site in struct:
        # 检查原子是否为氧
        if site.specie.symbol == "O":
            # 获取氧原子的标签（label）
            oxygen_labels.append(site.label)  # 假设标签存储在 "label" 属性中
    
    for oxygen_label in oxygen_labels:
        graph_results=graph_results+structure2graph(struct,oxygen_label,cif_name,0,0)
    
    return graph_results
        
        
        
        
    



def test(modeldic_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型权重
    heads = 4 
    hidden_dim = 64 # 隐藏特征单元
    batch_size =2 # 批处理大小
    lr =0.001  # 批处理大小0.0005
    dropout=0.3
    weight_decay=0.0001
    model = S5_model_GPU.GNNWithEdgeFeatures(num_node_features=32,num_edge_features=2,                                
                                hidden_dim=hidden_dim,heads=heads,
                                dropout=dropout,
                                nnconv_aggr="add")  # 节点特征维度为32
    model.load_state_dict(torch.load(modeldic_path,weights_only=False))
    model.to(device)
    model.eval()  # 设置模型为评估模式
    
    #遍历./test_cifs文件夹下的所有cif文件
    cif_paths=glob.glob("./test/temptest/*.cif")    #['./test_cifs\\Bi2Na1O11Sb3_ICSD_167071.cif', './test_cifs\\Bi4O7_mp-30303.cif',......]
    cif_datacsv_paths=glob.glob("./test/temptest/*.csv")
    ready_pre_n=len(cif_paths)-len(cif_datacsv_paths)
    print("晶体库cif数量：",len(cif_paths)," 已预测数量：",len(cif_paths)-ready_pre_n," 待预测数量：",ready_pre_n)
    
    
    for cif_path in cif_paths:
        cif_name=cif_path.replace("./test/temptest\\","").replace(".cif","")
        #判断某个晶体的预测数据是否已经存在
        if not  os.path.exists("./test/temptest/"+cif_name+".csv"):
            print("正在预测：",cif_name)
            sites_graph_data=trans_one_cif(cif_path,cif_name)
            this_cif_predata=[]
            for site_graph_data in sites_graph_data:
                y = torch.tensor(site_graph_data["Oxy_vacancy_formation_energy"], dtype=torch.float)
                x = torch.tensor(site_graph_data["node_features"], dtype=torch.float)
                source_nodes = [edge[0] for edge in site_graph_data["edges"]]
                target_nodes = [edge[1] for edge in site_graph_data["edges"]]
                edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
                edge_attr = torch.tensor(site_graph_data["edge_attr"], dtype=torch.float)
                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
                data = data.to(device)
                out = model(data)   # 前向传播
                out=out.cpu()
                #print(out.item())
                this_cif_predata.append([cif_name,site_graph_data["Oxy_site_label"],site_graph_data["Oxy_site_index"],out.item()])
            #将this_cif_predata写入csv文件
            this_cif_predata=pd.DataFrame(this_cif_predata,columns=["crystal_name","Oxy_site_label","Oxy_site_index","Oxy_vacancy_formation_energy"])
            this_cif_predata.to_csv("./test/temptest/"+cif_name+".csv",index=False)
            ready_pre_n=ready_pre_n-1
            print("晶体库cif数量：",len(cif_paths)," 已预测数量：",len(cif_paths)-ready_pre_n," 待预测数量：",ready_pre_n)
    
def test_statistics():
    cif_datacsv_paths=glob.glob("./test/temptest/*.csv")
    test_statistics_alllist=[]
    for this_cif_datacsv_path in cif_datacsv_paths:
        #使用pandas读取csv文件，无index列，第一行是表头
        this_cif_datacsv=pd.read_csv(this_cif_datacsv_path,index_col=None,header=0)
        this_cif_compoundname=this_cif_datacsv["crystal_name"][0].split("_")[0]
        # 对Oxy_vacancy_formation_energy列分别计算平均值、最小值、最大值、中位数、上四分中位数、下四分中位数、标准差
        this_mean=this_cif_datacsv["Oxy_vacancy_formation_energy"].mean()
        this_min=this_cif_datacsv["Oxy_vacancy_formation_energy"].min()
        this_max=this_cif_datacsv["Oxy_vacancy_formation_energy"].max()
        this_median=this_cif_datacsv["Oxy_vacancy_formation_energy"].median()
        this_25=this_cif_datacsv["Oxy_vacancy_formation_energy"].quantile(0.25)
        this_75=this_cif_datacsv["Oxy_vacancy_formation_energy"].quantile(0.75)
        this_std=this_cif_datacsv["Oxy_vacancy_formation_energy"].std()
        test_statistics_alllist.append([this_cif_compoundname,this_cif_datacsv["crystal_name"][0],
                                        round(this_mean,3) ,round(this_min,3), round(this_max,3), round(this_median,3),
                                        round(this_25,3), round(this_75,3), round(this_std,3)])
    #将test_statistics_alllist中的行，按照行中第三列的数值从小到大的顺序排列
    test_statistics_alllist.sort(key=lambda x: x[2])
    test_statistics=pd.DataFrame(test_statistics_alllist,columns=["化合物名称","cif文件名","平均值","最小值","最大值","中位数","下四分中位数","上四分中位数","标准差"])
    test_statistics.to_csv("./test/temptest/temp_statistics.csv",index=True)
    #print(this_cif_datacsv.values)


if __name__ == "__main__":
    modeldic_path=r'D:\program_202410_with_2dataset\result_GPU\train_with_alldata\trainloss0.32120604413087817_testmaeloss0.1892285860037502_modeldict.pth'
    test(modeldic_path)   #第一步。
    #test_statistics()    #第二步，使用前先删除旧的统计文件
