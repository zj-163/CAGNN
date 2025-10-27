import re
import numpy as np
import pandas as pd
import os


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


name_id_csv="./log_01_03_22.csv"
name_id_data = pd.read_csv(name_id_csv, index_col=None, header=0).values
name_id_data[:, 1] = name_id_data[:, 1].astype(str)  #将第二列转换为字符串
id_prop_csv="./id_prop.csv.all"
id_prop_data = pd.read_csv(id_prop_csv, index_col=None, header=None).values
charge0_data = pd.read_csv('../charge0.csv', index_col=0, header=0).values
#print("name_id_data:",name_id_data)
#print("id_prop_data:",id_prop_data)
#print("charge0_data:",charge0_data[:,30])
"""
#遍历name_id_data的第一列，检查是否在charge0_data的第一列
for i in range(len(name_id_data)):
    if name_id_data[i, 0] in charge0_data[:, 0]:
        print(name_id_data[i, 0],name_id_data[i, 1])
"""
# 遍历id_prop.csv.all，这是一个csv文件，如果它的第一列中含有“O”，则将这一行保存在新的列表中
listO=[]
for i in range(len(id_prop_data)):
    if "O" in id_prop_data[i, 0]:
        listO.append(id_prop_data[i, :])
#保存到新的csv文件
listO_df=pd.DataFrame(listO)
listO_df.to_csv("./listO.csv", index=False)


listO_data_withoutFe=[]
listO_data_withFe=[]
for i in range(len(name_id_data)):
    #使用extract_elements提取name_id_data[i,0]中的元素
    elements=extract_elements(name_id_data[i,0])
    #如果name_id_data[i,0]
    filenamestart=""
    if name_id_data[i,1].startswith("icsd"):
        filenamestart=name_id_data[i,1]
    else:
        if len(name_id_data[i,1]) < 7:
            zeros_needed = 7 - len(name_id_data[i,1]) # 计算需要补充的0的数量
            filenamestart = '0' * zeros_needed + name_id_data[i,1] # 在字符串前面补充足够的0
    
    #如果name_id_data[i,0]中含有Mn\Fe\Co\Ni
    if "Mn" in elements or "Fe" in elements or "Co" in elements or "Ni" in elements:
        #遍历id_prop_data，将id_prop_data中第一列以filenamestart起始，且含有"O"的行，保存在listO_data_withFe中
        for j in range(len(id_prop_data)):
            if id_prop_data[j, 0].startswith(filenamestart) and "O" in id_prop_data[j, 0]:
                listO_data_withFe.append(id_prop_data[j, :].tolist()+[name_id_data[i,0]])
    else:
        #遍历id_prop_data，将id_prop_data中第一列以filenamestart起始，且含有"O"的行，保存在listO_data_withoutFe中
        for j in range(len(id_prop_data)):
            if id_prop_data[j, 0].startswith(filenamestart) and "O" in id_prop_data[j, 0]:
                listO_data_withoutFe.append(id_prop_data[j, :].tolist()+[name_id_data[i,0]])

#将listO_data_withoutFe和listO_data_withFe保存到新的csv文件
listO_data_withoutFe_np=np.array(listO_data_withoutFe)
listO_data_withFe_np=np.array(listO_data_withFe)


listO_data_withoutFe_np[:, 1] = listO_data_withoutFe_np[:, 1].astype(float)
listO_data_withFe_np[:, 1] = listO_data_withFe_np[:, 1].astype(float)


listO_data_withoutFe_df=pd.DataFrame(listO_data_withoutFe_np)
listO_data_withFe_df=pd.DataFrame(listO_data_withFe_np)
listO_data_withoutFe_df.to_csv("./listO_data_withoutFe.csv", index=False)
listO_data_withFe_df.to_csv("./listO_data_withFe.csv", index=False)


#print(listO_data_withFe_np[:, 1].astype(float))




#分别对listO的第二列数据，和charge0的第30列数据，绘制箱线图，绘制成两个图最后拼接，不使用seaborn
listO_data = pd.read_csv('./listO.csv', index_col=None, header=0).values
charge0_data = pd.read_csv('../charge0.csv', index_col=0, header=0).values

listO_data[:, 1] = listO_data[:, 1].astype(float)
charge0_data[:, 30] = charge0_data[:, 30].astype(float)

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

a=listO_data_withoutFe_np[:, 1].astype(float)
#print("a",a.tolist())

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制箱线图
ax.boxplot([charge0_data[:, 30], listO_data[:, 1], listO_data_withoutFe_np[:, 1].astype(float), listO_data_withFe_np[:, 1].astype(float)], 
           labels=["dataset1", "dataset2", "dataset2 (without Mn/Fe/Co/Ni)", "dataset2 (with Mn/Fe/Co/Ni)"])

# 调整图形下边距
plt.subplots_adjust(bottom=0.2)  # 根据需要调整这个值
# 设置 x 轴标签向右下偏斜 40 度
plt.xticks(rotation=-20)

plt.show()
# 以svg格式保存图形 
fig.savefig("boxplot.svg", format="svg")



# 分别对下面的数据绘制正态分布图，并以2*2放在画布上 charge0_data[:, 30], listO_data[:, 1], listO_data_withoutFe_np[:, 1].astype(float), listO_data_withFe_np[:, 1].astype(float)

import seaborn as sns

# 创建一个2x2的画布
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 绘制第一个正态分布图
sns.histplot(charge0_data[:, 30], kde=True, ax=axs[0, 0])
axs[0, 0].set_title('dataset1')
axs[0, 0].set_xlabel('oxygen vacancy formation energy (eV)')  # 添加x轴标签和单位
axs[0, 0].set_ylabel('sample count')  # 添加y轴标签和单位

# 绘制第二个正态分布图
sns.histplot(listO_data[:, 1], kde=True, ax=axs[0, 1])
axs[0, 1].set_title('dataset2')
axs[0, 1].set_xlabel('oxygen vacancy formation energy (eV)')  # 添加x轴标签和单位
axs[0, 1].set_ylabel('sample count')  # 添加y轴标签和单位


# 绘制第三个正态分布图
sns.histplot(listO_data_withoutFe_np[:, 1].astype(float), kde=True, ax=axs[1, 0])
axs[1, 0].set_title('dataset2 (without Mn/Fe/Co/Ni)')
axs[1, 0].set_xlabel('oxygen vacancy formation energy (eV)')  # 添加x轴标签和单位
axs[1, 0].set_ylabel('sample count')  # 添加y轴标签和单位


# 绘制第四个正态分布图
sns.histplot(listO_data_withFe_np[:, 1].astype(float), kde=True, ax=axs[1, 1])
axs[1, 1].set_title('dataset2 (with Mn/Fe/Co/Ni)')
axs[1, 1].set_xlabel('oxygen vacancy formation energy (eV)')  # 添加x轴标签和单位
axs[1, 1].set_ylabel('sample count')  # 添加y轴标签和单位


# 调整布局
plt.tight_layout(h_pad=3, w_pad=2)

# 显示图形
plt.show()
# 以svg格式保存图形 
fig.savefig("histogram.svg", format="svg")


"""
creat_cif_folder=".\\cifs"
temp_folder=".\\cifs\\temp"
allcifname_list=[f for f in os.listdir(creat_cif_folder) if os.path.isfile(os.path.join(creat_cif_folder, f))]
# 以上定义变量要与trans_graph_data_to_file一致
cif_filename=allcifname_list[i]
cif_number=""
if cif_filename.startswith("icsd"):
    cif_number=allcifname_list[i].split("-")[0]
else:
    cif_number=str(int(allcifname_list[i].split("-")[0]))

name_id_csv="./log_01_03_22.csv"
name_id_data = pd.read_csv(name_id_csv, index_col=None, header=0).values
name_id_data[:, 1] = name_id_data[:, 1].astype(str)
id_prop_csv="./id_prop.csv.all"
id_prop_data = pd.read_csv(id_prop_csv, index_col=None, header=None).values

this_name=name_id_data[name_id_data[:, 1] == cif_number, 0][0]

cif_path=creat_cif_folder+"\\"+cif_filename

struct=None
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    # 在这个块中的代码会忽略用户警告,用于屏蔽晶体坐标读取四舍五入的通知
    struct= Structure.from_file(cif_path)
this_O=cif_filename.split("-")[1].split(".")[0]
this_y=float(id_prop_data[id_prop_data[:, 0] == cif_filename.split(".")[0], 1][0])
results=structure2graph(struct,this_O,this_name,i+1,this_y)
print(i)
with open(temp_folder+"\\"+str(i+1)+".json", 'w', encoding='utf-8') as file:
    json.dump(results, file, ensure_ascii=False, indent=4)
"""