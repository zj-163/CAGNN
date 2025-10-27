import numpy as np
import pandas as pd
import random
import json
from collections import defaultdict
import re
import networkx as nx

# 本程序用于将化合物按照含有的阳离子数量进行拆分，并保存为csv文件


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

# 检查含有不同阳离子数量的化合物数量
def check_elements_numbers():
    """
    ele_1: 28
    ele_2: 499
    ele_3: 286
    ele_4: 11
    ele_5: 0
    ele_6: 0
    ele_7: 0
    ele_8: 0
    """
    ov_0_csvpath="../charge0_with_dataset2.csv"
    dft_data = pd.read_csv(ov_0_csvpath, index_col=None, header=0).values
    # 获取dft_data的第1列，即化合物名称 
    compound_list=dft_data[:,1]
    # 使用集合对compound_list进行去重
    compound_list_unique = list(set(compound_list))
    random.seed(72)
    random.shuffle(compound_list_unique)
    compounds = compound_list_unique

    # 创建一个字典来存储每个元素对应的化合物
    element_to_compounds = defaultdict(list)
    ele_1=[]
    ele_2=[]
    ele_3=[]
    ele_4=[]
    ele_5=[]
    ele_6=[]
    ele_7=[]
    ele_8=[]
    for compound in compounds:
        elements = extract_elements(compound)
        print(elements)
        if len(elements) == 1:
            ele_1.append(compound)
        elif len(elements) == 2:
            ele_2.append(compound) 
        elif len(elements) == 3:
            ele_3.append(compound)
        elif len(elements) == 4:
            ele_4.append(compound)
        elif len(elements) == 5:
            ele_5.append(compound)
        elif len(elements) == 6:
            ele_6.append(compound)
        elif len(elements) == 7:
            ele_7.append(compound)
        elif len(elements) == 8:
            ele_8.append(compound)

    print("ele_1:", len(ele_1))
    print("ele_2:", len(ele_2))
    print("ele_3:", len(ele_3))
    print("ele_4:", len(ele_4))
    print("ele_5:", len(ele_5))
    print("ele_6:", len(ele_6))
    print("ele_7:", len(ele_7))
    print("ele_8:", len(ele_8))

if __name__ == "__main__":
    ov_0_csvpath="../charge0_with_dataset2.csv"
    dft_data = pd.read_csv(ov_0_csvpath, index_col=None, header=0).values
    # 获取dft_data的第1列，即化合物名称 
    compound_list=dft_data[:,1]
    # 使用集合对compound_list进行去重
    compound_list_unique = list(set(compound_list))
    random.seed(72)
    random.shuffle(compound_list_unique)
    compounds = compound_list_unique

    # 创建一个字典来存储不同阳离子数量对应的化合物
    element_numbers_compounds_dict = defaultdict(list)

    for compound in compounds:
        elements = extract_elements(compound)
        #print(elements)
        if len(elements) == 1:
            element_numbers_compounds_dict["eles_1"].append(compound)
        elif len(elements) == 2:
            element_numbers_compounds_dict["eles_2"].append(compound)
        elif len(elements) == 3:
            element_numbers_compounds_dict["eles_3"].append(compound)
        elif len(elements) == 4:
            element_numbers_compounds_dict["eles_4"].append(compound)


    for key, value in element_numbers_compounds_dict.items():
        dft_data_subset=dft_data[np.isin(dft_data[:,1], value)]
        dft_data_subset_df=pd.DataFrame(dft_data_subset, columns= pd.read_csv(ov_0_csvpath, index_col=None, header=0).columns)
        dft_data_subset_df.to_csv(f"./split_by_elements_numbers/{key}.csv", index=False)

    
    

    

