import re
import shutil
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


if __name__ == '__main__':
    name_id_csv="./log_01_03_22.csv"
    name_id_data = pd.read_csv(name_id_csv, index_col=None, header=0).values
    name_id_data[:, 1] = name_id_data[:, 1].astype(str)  #将第二列转换为字符串
    id_prop_csv="./id_prop.csv.all"
    id_prop_data = pd.read_csv(id_prop_csv, index_col=None, header=None).values
    charge0_data = pd.read_csv('./charge0.csv', index_col=0, header=0).values

    list_d2=[]
    for i in range(len(name_id_data)):
        filenamestart=""
        if name_id_data[i,1].startswith("icsd"):
            filenamestart=name_id_data[i,1]
        else:
            if len(name_id_data[i,1]) < 7:
                zeros_needed = 7 - len(name_id_data[i,1]) # 计算需要补充的0的数量
                filenamestart = '0' * zeros_needed + name_id_data[i,1] # 在字符串前面补充足够的0
        
        #遍历id_prop_data，将id_prop_data中第一列以filenamestart起始，且含有"O"的行，保存在listO_data_withFe中
        for j in range(len(id_prop_data)):
            if id_prop_data[j, 0].startswith(filenamestart) and "O" in id_prop_data[j, 0]:
                    list_d2.append( [name_id_data[i,0]+"-dataset2-"+filenamestart,
                                    name_id_data[i,0]+"-dataset2-"+filenamestart+"_Va_"+id_prop_data[j, 0].replace(filenamestart+"-",""),
                                    id_prop_data[j, 1]         ]   )
        
        # 将cifs文件夹中的filenamestart+“-O1.cif”复制到cif整理文件夹中，命名为name_id_data[i,0]+"-dataset2-"+filenamestart+".cif"
        if os.path.exists("./cifs/"+filenamestart+"-O1.cif"):
            shutil.copyfile("./cifs/"+filenamestart+"-O1.cif", "./cif整理/"+name_id_data[i,0]+"-dataset2-"+filenamestart+".cif")
        else:
            print("cifs文件夹中不存在"+filenamestart+"-O1.cif")
            #注意，Ce1O4Sr2-dataset2-0290198，需要手动复制重命名，因为只有O2

    #print(list_d2)

    #将list_d2追加到charge0_data中，   list_d2[:,0]=charge0_data[:,0],list_d2[:,1]=charge0_data[:,1],list_d2[:,2]=charge0_data[:,30],其他列留空
    charge0_data_withdataset2 = np.append(charge0_data, np.zeros((len(list_d2), charge0_data.shape[1])), axis=0)
    for i in range(len(list_d2)):
        charge0_data_withdataset2[charge0_data_withdataset2.shape[0]-len(list_d2)+i, 0] = list_d2[i][0]
        charge0_data_withdataset2[charge0_data_withdataset2.shape[0]-len(list_d2)+i, 1] = list_d2[i][1]
        charge0_data_withdataset2[charge0_data_withdataset2.shape[0]-len(list_d2)+i, 30] = list_d2[i][2]

    #将charge0_data保存为charge0.csv
    pd.DataFrame(charge0_data_withdataset2).to_csv('./charge0_with_dataset2.csv', index=True, header=pd.read_csv('./charge0.csv', index_col=0, header=0).columns)
    #将list_d2保存为dataset2.csv
    pd.DataFrame(list_d2).to_csv('./dataset2.csv', index=False, header=False)



