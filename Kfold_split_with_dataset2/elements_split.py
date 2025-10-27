import numpy as np
import pandas as pd
import random
import json
from collections import defaultdict
import re
import networkx as nx

# 本程序用于将化合物按照元素进行拆分，并保存为csv文件，每个阳离子单独做一份交叉验证

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

    # 创建一个字典来存储每个元素对应的化合物
    element_to_compounds = defaultdict(list)

    for compound in compounds:
        elements = extract_elements(compound)
        for element in elements:
            element_to_compounds[element].append(compound)

    print(element_to_compounds)
    # 将index_list保存为json文件
    with open("./split_by_elements/element_to_compounds.json", "w") as f:
        json.dump(element_to_compounds, f,ensure_ascii=False, indent=4)

    for key, value in element_to_compounds.items():
        dft_data_subset=dft_data[np.isin(dft_data[:,1], value)]
        dft_data_subset_df=pd.DataFrame(dft_data_subset, columns= pd.read_csv(ov_0_csvpath, index_col=None, header=0).columns)
        dft_data_subset_df.to_csv(f"./split_by_elements/{key}_y.csv", index=False)

        dft_data_subset=dft_data[~np.isin(dft_data[:,1], value)]   # ~np.isin(dft_data[:,1], value)：取反
        dft_data_subset_df=pd.DataFrame(dft_data_subset, columns= pd.read_csv(ov_0_csvpath, index_col=None, header=0).columns)
        dft_data_subset_df.to_csv(f"./split_by_elements/{key}_x.csv", index=False)

    

