import json
import pandas as pd
from tqdm import tqdm
import os
from collections import defaultdict
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
"""
独特氧位点： 2540
独特化学式： 1023
所有氧位点数量： 14938
"""

def 空间群识别():
    # 定义文件夹路径
    folder_path = 'transformed_cif_with_dataset2'

    # 使用 defaultdict 记录每个空间群的文件名
    spacegroup_dict = defaultdict(list)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.cif'):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 使用 pymatgen 读取 CIF 文件
                structure = Structure.from_file(file_path)
                
                # 使用 SpacegroupAnalyzer 分析空间群
                analyzer = SpacegroupAnalyzer(structure)
                space_group = analyzer.get_space_group_symbol()
                
                # 将文件名添加到对应空间群的列表中
                spacegroup_dict[space_group].append(filename)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    # 输出结果
    for space_group, filenames in spacegroup_dict.items():
        print(f"空间群: {space_group}, 晶体数量: {len(filenames)}")
        print(f"文件名列表: {filenames}")
        print("-" * 50)

    # 统计总共有多少种空间群
    total_space_groups = len(spacegroup_dict)
    print(f"总共有 {total_space_groups} 种不同的空间群")

    # 将空间群为第一列，对应空间群的晶体文件名列表合并为一个文本为第二列，晶体数量为第3列，存储到一个csv文件中
    # 创建一个空的 DataFrame
    df = pd.DataFrame(columns=['space_group', 'filenames', 'count'])

    # 遍历 spacegroup_dict，将数据添加到 DataFrame 中
    for space_group, filenames in spacegroup_dict.items():
        # 将文件名列表合并为一个文本字符串
        filenames_text = ', '.join(filenames)
        # 使用 pd.concat 代替已弃用的 df.append
        df = pd.concat([df, pd.DataFrame({
            'space_group': [space_group],
            'filenames': [filenames_text],
            'count': [len(filenames)]
        })], ignore_index=True)

    # 保存到 CSV 文件
    df.to_csv('./dataset2/space_group_with_dataset2.csv', index=True)

if __name__ == "__main__":
    
    # 使用pandas读取charge0_with_2dataset.csv
    df = pd.read_csv('./charge0_with_dataset2.csv')
    # 统计信息
    所有未去重化学式list=df["formula"].values.tolist()
    print("独特氧位点：",len(所有未去重化学式list))
    print("独特化学式：",len(list(set(所有未去重化学式list))))

    """
    # 所有化学式
    graph_data_path="./transformed_graph_with_dataset2/"
    dft_dataframe_x = pd.read_csv('./charge0_with_dataset2.csv')

    所有氧位点数量 = 0
    #为下面的循环添加tqdm进度条
    for vacancy_data in tqdm(dft_dataframe_x.values,colour="green",desc="Loading x data"):
        with open(graph_data_path+vacancy_data[2]+".json", 'r', encoding='utf-8') as file:
            this_vacancy_graphdata_list = json.load(file)
            所有氧位点数量 += len(this_vacancy_graphdata_list)
    print("所有氧位点数量：",所有氧位点数量)
    
    空间群识别()
    """
