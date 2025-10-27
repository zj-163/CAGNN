

# 遍历./split_by_elements下的文件夹
import os


统计数组=[]
for folder in os.listdir("./split_by_elements"):
    print(folder)
    # 遍历文件夹下的以modeldict.pth结尾的文件的文件名
    for file in os.listdir(f"./split_by_elements/{folder}"):
        if file.endswith("modeldict.pth"):
            # 提取文件名中trainloss和_testmaeloss中间的部分，转换为浮点数
            trainloss = float(file.split("trainloss")[1].split("_testmaeloss")[0])
            # 提取文件名中testmaeloss和_modeldict中间的部分，转换为浮点数
            testmaeloss = float(file.split("testmaeloss")[1].split("_modeldict")[0])
            # 将“split_by_elements”、文件夹名、trainloss和testmaeloss添加到统计数组中
            统计数组.append(["split_by_elements", folder, trainloss, testmaeloss])

# 将统计数组转换为DataFrame，并保存为CSV文件
import pandas as pd
df = pd.DataFrame(统计数组, columns=["split_type", "fold", "trainloss", "test_loss"])
df.to_csv("./split_by_elements/allresult.csv", index=False)

    

