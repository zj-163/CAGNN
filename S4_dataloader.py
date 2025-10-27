# _with_dataset2

import json
import pandas as pd
import numpy as np
import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
from tqdm import tqdm
def read_graph_data_from_file(batch_size=2,ov_0_x_csvpath_list=[],ov_0_y_csvpath_list=[]):
    graph_data_path="./transformed_graph_with_dataset2/"
    print("waiting for loading csv data......")
    dft_dataframe_x = pd.concat([pd.read_csv(file, index_col=None, header=0) for file in ov_0_x_csvpath_list])
    dft_dataframe_y = pd.concat([pd.read_csv(file, index_col=None, header=0) for file in ov_0_y_csvpath_list])
    
    train_dataset = []
    #为下面的循环添加tqdm进度条
    for vacancy_data in tqdm(dft_dataframe_x.values,colour="green",desc="Loading x data"):
        with open(graph_data_path+vacancy_data[2]+".json", 'r', encoding='utf-8') as file:
            this_vacancy_graphdata_list = json.load(file)
        for site_graph_data in this_vacancy_graphdata_list:
            """
            print("x, crystal_name:",site_graph_data["crystal_name"], 
                  "Oxy_site_label:",site_graph_data["Oxy_site_label"],
                  "Oxy_site_index:",site_graph_data["Oxy_site_index"],
                  "vacancy_formation_energy:",site_graph_data["Oxy_vacancy_formation_energy"])
            """
            y = torch.tensor(site_graph_data["Oxy_vacancy_formation_energy"], dtype=torch.float)
            x = torch.tensor(site_graph_data["node_features"], dtype=torch.float)
            source_nodes = [edge[0] for edge in site_graph_data["edges"]]
            target_nodes = [edge[1] for edge in site_graph_data["edges"]]
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            edge_attr = torch.tensor(site_graph_data["edge_attr"], dtype=torch.float)
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            train_dataset.append(data)

    test_dataset = []
    for vacancy_data in tqdm(dft_dataframe_y.values,colour="green",desc="Loading y data"):
        with open(graph_data_path+vacancy_data[2]+".json", 'r', encoding='utf-8') as file:
            this_vacancy_graphdata_list = json.load(file)
        for site_graph_data in this_vacancy_graphdata_list:
            """
            print("y, crystal_name:",site_graph_data["crystal_name"], 
                  "Oxy_site_label:",site_graph_data["Oxy_site_label"],
                  "Oxy_site_index:",site_graph_data["Oxy_site_index"],
                  "vacancy_formation_energy:",site_graph_data["Oxy_vacancy_formation_energy"])
            """
            y = torch.tensor(site_graph_data["Oxy_vacancy_formation_energy"], dtype=torch.float)
            x = torch.tensor(site_graph_data["node_features"], dtype=torch.float)
            source_nodes = [edge[0] for edge in site_graph_data["edges"]]
            target_nodes = [edge[1] for edge in site_graph_data["edges"]]
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            edge_attr = torch.tensor(site_graph_data["edge_attr"], dtype=torch.float)
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            test_dataset.append(data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader



def test():
    ov_0_x_csvpath_list=["./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_1.csv",
                         "./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_2.csv",
                         "./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_3.csv",
                         "./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_4.csv"]
    ov_0_y_csvpath_list=["./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_0.csv"]
    dft_dataframe_x = pd.concat([pd.read_csv(file, index_col=None, header=0) for file in ov_0_x_csvpath_list])
    dft_dataframe_y = pd.concat([pd.read_csv(file, index_col=None, header=0) for file in ov_0_y_csvpath_list])
    print(len(dft_dataframe_x.values))
    print(len(dft_dataframe_y.values))
    print(dft_dataframe_x.values[0,2])  #KNaLaNbO5_Va_O1


if __name__ == "__main__":
    ov_0_x_csvpath_list=["./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_1.csv",
                         "./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_2.csv",
                         "./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_3.csv",
                         "./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_4.csv"]
    ov_0_y_csvpath_list=["./Kfold_split_with_dataset2/split_by_vacancies/dft_data_subset_0.csv"]
    train_loader,test_loader=read_graph_data_from_file(2,ov_0_x_csvpath_list,ov_0_y_csvpath_list)
    print(len(train_loader))
    print(len(test_loader))
