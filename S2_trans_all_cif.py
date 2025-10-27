import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from S1_supercell_to_std import supercell2std
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

def creat_cif():
    # 使用 pd.read_csv() 函数读取CSV文件
    df = pd.read_csv('./charge0.csv', index_col=0, header=0)

    name_column =np.unique( df.iloc[:, 0].to_numpy()) # 使用iloc和列索引获取整列,转换为numpy数组并去重
    #print(name_column)
    
    home_path="G:\\O2p_Paper\\program_202410\\site_info\\"
    creat_cif_folder="G:\\O2p_Paper\\program_202410\\transformed_cif\\"

    for i in tqdm(range(len(name_column))):
        # 构造文件路径
        this_name=name_column[i]
        supercell_path=home_path+this_name+"\\supercell.cif"
        cell_info_path=home_path+this_name+"\\cell_info.txt"
        creat_cif_path=creat_cif_folder+this_name+".cif"
        # 检查文件是否存在
        assert os.path.exists(supercell_path)
        assert os.path.exists(cell_info_path)
        this_structure=supercell2std( supercell_path, cell_info_path)
        this_structure.to_file(creat_cif_path)


if __name__ == '__main__':
    creat_cif()