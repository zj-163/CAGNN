# CAGNN
This is the code repository for the "_Prediction of Oxygen Vacancy Formation Energy in Oxides Based on the Graph Neural Network Approach_" paper.

## 1、Data source:
- dataset1：  Insights into oxygen vacancies from high-throughput first-principles calculations (DOI: 10.1103/PhysRevMaterials.5.123803)
- dataset2：  Defect graph neural networks for materials discovery in high-temperature clean-energy applications (DOI: 10.1038/s43588-023-00495-2)

## 2、Environment configuration
- All algorithms were implemented in Python 3.12.7 using the Pytorch 2.4.1 (Cuda 11.8) framework and were carried out on a computer with an AMD Ryzen 9 7950X CPU, an Nvidia RTX 3090Ti GPU, and 64GB RAM. 



## 3、Project description
- siteinfo:  It includes the cellinfo.txt and supercell.cif files for each crystal in dataset1.
- dataset2:  Store the data of dataset2 , rename the cifs, and merge it with the data of dataset1 to generate charge0_with_dataset2.csv
- python_environments_GPU: Python environment for the project.
- Kfold_split_with_dataset2: K-fold cross-validation data (dataset1 & dataset2), corresponding to Sections 2.1.2 and 2.1.3 of the manuscript.
- result_GPU：Model training results based on different K-fold cross-validation data.
- S8_compare_data： model result of cgcnn and ovgnn_only_structure, corresponding to Table 1 of the manuscript.
- transformed_cif_with_dataset2：The result of converting the CIF files in two datasets into standard unit cells with O-site labels
- transformed_graph_with_dataset2：The result of converting the crystal files and the corresponding oxygen vacancy formation energy data in "transformed_cif_with_dataset2" into the input data for the graph neural network.
- test: the data corresponding to Sections 3.4 of the manuscript.
- charge0_with_dataset2.csv:  It includes the formation energies of each oxygen site of all crystals in the two datasets.
- Elements_props_all_v1.csv:  The properties that encompass all the elements. The data is sourced from the pymatgen.
- S0_statistic_informations_of_two_datasets.py: Used for collecting crystal information in the dataset
- S1_supercell_to_std.py:  Convert the supercell to a standard unit cell
- S2_trans_all_cif.py:  Batch conversion of all cif files in the "siteinfo" folder to standard unit cell stdcif format.
- S3.2_stdcif_to_graphdata_with_dataset2.py：Read the CIF file and convert it into graph data (dataset1 & dataset2).
- S4_dataloader.py:  Read the graph data files of two datasets and convert them into a dataloader
- S5_model_GPU.py:  Train the model using Kfold_split_with_dataset2
- S5.3_model_GPU_testCV.py:  Test the model using Kfold_split_with_dataset2
- S6.1_temp_test.py：Perform predictions on the cif files in the "test/temptest" folder
- S8_x:  model training and test code of cgcnn and ovgnn_only_structure, corresponding to Table 1 of the manuscript.




