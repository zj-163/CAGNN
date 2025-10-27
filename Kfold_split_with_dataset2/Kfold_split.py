import numpy as np
import pandas as pd
import random
import json
from collections import defaultdict
import re

#---------按氧空位划分---------
def split_by_vacancies():
    # 1. 读取文件charge0.csv
    ov_0_csvpath="../charge0_with_dataset2.csv"
    dft_data = pd.read_csv(ov_0_csvpath, index_col=None, header=0).values
    # 输出二维数据中第二列的数据
    # print(dft_data[:,1])
    # 将dft_data按顺序均匀划分为5份
    dft_data_sequences_list=[[],[],[],[],[]]   
    list_i=0 
    for data in dft_data:
        dft_data_sequences_list[list_i].append(data)
        list_i=(list_i+1)%5

    # 将dft_data_shuffle_list转换为5个pd.DataFrame,header与charge0.csv一致
    for i in range(5):
        dft_data_sequences_list[i]=pd.DataFrame(dft_data_sequences_list[i], columns= pd.read_csv(ov_0_csvpath, index_col=None, header=0).columns)
        # 保存为csv文件
        dft_data_sequences_list[i].to_csv(f"./split_by_vacancies/dft_data_subset_{i}.csv", index=False)


#---------按化合物划分---------
def split_by_compound():
    # 1. 读取文件charge0.csv
    ov_0_csvpath="../charge0_with_dataset2.csv"
    dft_data = pd.read_csv(ov_0_csvpath, index_col=None, header=0).values
    # 获取dft_data的第1列，即化合物名称 
    compound_list=dft_data[:,1]
    print(compound_list)
    # 使用集合对compound_list进行去重
    compound_list_unique = list(set(compound_list))
    print(len(compound_list_unique))
    random.seed(42)
    random.shuffle(compound_list_unique)
    compound_list_unique_kfold=np.array_split(compound_list_unique, 5)
    for i in range(5):
        # 遍历dft_data，将第1列在compound_list_unique_kfold[i]中的行保存为一个新的csv文件
        dft_data_subset=dft_data[np.isin(dft_data[:,1], compound_list_unique_kfold[i])]
        dft_data_subset_df=pd.DataFrame(dft_data_subset, columns= pd.read_csv(ov_0_csvpath, index_col=None, header=0).columns)
        dft_data_subset_df.to_csv(f"./split_by_compounds/dft_data_subset_{i}.csv", index=False)

    # 将index_list保存为json文件
    with open("./split_by_compounds/compound_list_unique.json", "w") as f:
        json.dump(compound_list_unique, f,ensure_ascii=False, indent=4)

if __name__ == "__main__":
    split_by_compound()
    split_by_vacancies()
