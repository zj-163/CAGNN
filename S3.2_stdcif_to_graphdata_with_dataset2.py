import json
import pandas as pd
import numpy as np
import os
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.analysis.local_env import NearNeighbors,BrunnerNN_real,IsayevNN,ValenceIonicRadiusEvaluator
from multiprocessing import Pool
import warnings


def structure2graph(this_structure,this_O_label,this_name,crystal_index,y):
    """
    
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

                if i_atom_0==i_atom_1:     # 判断是否为特定O空位,数组索引为12
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


# 创建线程任务函数
def for_task(i):
    transformed_cif_folder="./transformed_cif_with_dataset2"
    graph_folder="./transformed_graph_with_dataset2"
    ov_0_csvpath="./charge0_with_dataset2.csv"
    dft_data = pd.read_csv(ov_0_csvpath, index_col=None, header=0).values
    
    this_name=dft_data[i][1]
    cif_path=transformed_cif_folder+"\\"+this_name+".cif"
    struct=None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # 在这个块中的代码会忽略用户警告,用于屏蔽晶体坐标读取四舍五入的通知
        struct= Structure.from_file(cif_path)
    this_O=dft_data[i][2].replace(dft_data[i][1]+"_Va_","")
    results=structure2graph(struct,this_O,this_name,i+1,dft_data[i][31])
    print(i)
    with open(graph_folder+"\\"+dft_data[i][2]+".json", 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    return i

def trans_graph_data_to_file():
    transformed_cif_folder="./transformed_cif_with_dataset2"
    ov_0_csvpath="./charge0_with_dataset2.csv" 
    data_all=[]
    dft_data = pd.read_csv(ov_0_csvpath, index_col=None, header=0).values   #  dft_data[1744][1]  :ZrSiO4；  dft_data[0][1]  :Ag2BiO3
    #print(dft_data[1744][2])

    # 创建事项列表
    items =  range(len(dft_data))   # range(0, 1745)
    
    
    # 创建一个进程池
    with Pool(4) as pool:
        # 使用map方法处理数据项并收集结果
        results_pool = pool.map(for_task, items)

    """
    # 遍历临时文件夹，读取并保存数据
    temp_folder="G:\\O2p_Paper\\program\\my_cif_col\\temp2"
    # 遍历目录下的每个文件和文件夹
    for filename in os.listdir(temp_folder):
        # 构造完整的文件路径
        file_path = os.path.join(temp_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            this_list = json.load(file)
            for result in this_list:
                data_all.append(result)

    print("读取已完成，正在排序并保存数据......")
    # 排序并保存数据
    data_all.sort(key=lambda x: x['crystal_index'])
    with open('G:\\O2p_Paper\\program\\my_cif_col\\fulldata_list_with_y.json', 'w', encoding='utf-8') as file:
        json.dump(data_all, file, ensure_ascii=False, indent=4)
    """
        

if __name__ == "__main__":
    trans_graph_data_to_file()
    
