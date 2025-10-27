import ast
import numpy as np
from pymatgen.io.cif import CifParser
import spglib
from pymatgen.core import Structure, Lattice

def get_cell_info(cell_info_path):
    """
    获取cell_info.txt的信息
    ### Returns:(bravais_Lattice,transformation_matrix,sites)
    transformation_matrix:   3*3, [[3, 0, 0], [0, 3, 0], [0, 0, 2]]
    sites:  [   [sitelabel,[(start,end),,...],oxidation_state],...]  
     (start,end) is contained,such as: [['Ag1', [[0, 17], [72, 125]], 1],...]
    """
    sites=[]    # 用于存储维科夫位点
    transformation_matrix=[]    # 用于存储晶胞的变换矩阵
    bravais_Lattice=""    #用于存储晶胞的布拉维格子类型
    with open(cell_info_path, 'r') as file:
        lines = file.readlines()
        # 使用ast.literal_eval()来将字符串转换为列表
        transformation_matrix=[ast.literal_eval(i) for i in lines[1].split(": ")[1].split("  ")]
        bravais_Lattice= lines[0].split(": ")[1][0]
        i = 0
        while i < len(lines):
            if lines[i].startswith('   Irreducible element'):
                # 连续提取下面的几行，直到读取到空行为止
                label = lines[i].split(": ")[1].strip()
                i += 5    #到Equivalent atoms行
                tempstr=lines[i].split(": ")[1]
                start_ends=[]
                for ii in tempstr.split(" "):
                    start=int(ii.split("..")[0])
                    end=int(ii.split("..")[1])
                    start_ends.append((start,end))
                i += 3    #到Oxidation states行
                oxidation_state=int(lines[i].split(": ")[1])
                sites.append([label,start_ends,oxidation_state])
                i += 1
            else:
                i += 1
    return bravais_Lattice,transformation_matrix,sites




def supercell2std(supercell_path, cell_info_path):
    """
    将超晶胞转换为标准惯用胞（std）
    读取cell_info.txt文件，获取sites信息；
    读取supercell_path文件，获取超晶胞结构；
    return: std_structure(pymatgen.Structure)
    """
    # 1. 读取supercellCIF文件和cell_info.txt文件，
    # 将supercellCIF文件转换为pymatgen的结构
    bravais_Lattice,transformation_matrix,sites=get_cell_info(cell_info_path)
    parser = CifParser(supercell_path)   # 解析cif文件,from pymatgen.io.cif import CifParser
    structure = parser.parse_structures()[0]  # 获取第一个结构
   
    # 2. 将pymatgen的结构转换为spglib的格式
    lattice = structure.lattice.matrix # 晶格常数
    positions = structure.frac_coords # 分数坐标
    numbers = [site.specie.number for site in structure] # 原子种类
    cell = (lattice, positions, numbers) # 晶体结构
    #print(cell)

    # 3. 使用spglib识别转换
    symmetry_dataset = spglib.get_symmetry_dataset(cell,symprec=0.1,angle_tolerance=5)
    mapping_to_primitive=symmetry_dataset.mapping_to_primitive
    std_mapping_to_primitive=symmetry_dataset.std_mapping_to_primitive
    std_lattice=symmetry_dataset.std_lattice
    std_positions=symmetry_dataset.std_positions
    std_numbers=symmetry_dataset.std_types
    #识别mapping_to_primitive这个np整数数组中最大的那个数
    primitive_atom_num=np.max(mapping_to_primitive)
    primitive_atom_num=primitive_atom_num+1
    # 创建一个长度为primitive_atom_num的列表，用于存储原子种类
    primitive_labels=[]
    #print("mapping_to_primitive:",mapping_to_primitive)
    #print("std_mapping_to_primitive:",std_mapping_to_primitive)
    for i in range(primitive_atom_num):
        primitive_labels.append("")
    for input_i in range(len(mapping_to_primitive)):
        if primitive_labels[mapping_to_primitive[input_i]] == "": 
            #如果mapping_to_primitive中该位置对应的primitive_labels尚为空，则填补            
            for site in sites:
                for rng in site[1]:
                    if rng[0] <= input_i <= rng[1]:
                        primitive_labels[mapping_to_primitive[input_i]]=site[0]
                        break
    #print("primitive_labels:",primitive_labels)
    std_labels=[]
    for std_map_i in std_mapping_to_primitive:
        std_labels.append(primitive_labels[std_map_i]) 
    #print("std_labels:",std_labels)       
    std_structure=Structure(std_lattice,std_numbers,std_positions,labels=std_labels)
    #std_structure.to(filename="std_structure2.cif" ,fmt="cif")
    return std_structure

if __name__ == "__main__":
    """ cell_info_path="cell_info.txt"
    bravais_Lattice,transformation_matrix,sites=get_cell_info(cell_info_path)
    print(bravais_Lattice)
    print(transformation_matrix)
    print(sites) """

    supercell_path="supercell.cif"
    cell_info_path="cell_info.txt"
    std_structure=supercell2std(supercell_path, cell_info_path)
    std_structure.to(filename="std_structure.cif" ,fmt="cif")

