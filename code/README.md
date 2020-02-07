# HGMN: Hierarchical Graph Matching Networks for Deep Graph Similarity Learning

This is the repo for **Hierarchical Graph Matching Networks for Deep Graph Similarity Learning (under review)**.

In order to compute the graph similarity between any pair of graph-structured objects, HGMN can be use to jointly learn graph representations and a graph matching metric function for computing graph similarity in an end-to-end fashion. 
In particular, HGMN consists of a multi-perspective node-graph matching network for effectively learning cross-level interactions between parts of a graph and a whole graph, and
 a siamese graph neural network for learning global-level interactions between two graphs. 

[add the figure in paper ?]

# Run the codes

## 1. Requirements
Make sure you have installed all of following packages or libraries (including dependencies if necessary) in you machine:
1. `networkx        2.3`
2. `numpy           1.17.1`
3. `scikit-learn    0.21.3`
4. `scipy           1.3.1`
5. `torch           1.1.0`
6. `torch-geometric 1.3.1`
7. `torchvision     0.3.0`


## 2. Usages

### 2.1 classification task
(1) Prepare the dataset for classification tasks

Download the datasets from [this repo](https://github.com/runningoat/hgmn_dataset) and then put them into `dataset/CFG/`.

A sample directory structure:
```
├── dataset
│   ├── CFG
│   │   ├── ffmpeg_6ACFG_min20_max200
│   │   ├── ffmpeg_6ACFG_min3_max200
│   │   ├── ffmpeg_6ACFG_min50_max200
│   │   ├── OpenSSL_6ACFG_min20_max200
│   │   ├── OpenSSL_6ACFG_min3_max200
│   │   └── OpenSSL_6ACFG_min50_max200

```


(2) Specify some hyper-parameters for classification tasks in `src/cfg_config.py`

(3) Train and test the model by running the following command:

`python cfg_train.py`

### 2.2 regression task
(1) Prepare the dataset for regression tasks

Data for regression task are placed in `/dataset/GED` directory. All the files required by our codes can be downloaded following instructions in this [repo](https://github.com/yunshengb/SimGNN).
Please make sure you have downloaded all the 3 directories required by our code: `data`, `save`, and `result`.

After downloading these files, please put them under `/dataset/GED`, which is the default data folder by our configuration, or you can also specify your own data directory.

An example directory structure is:
``` 
dataset
└── GED
    ├── data
    │   ├── AIDS700nef
    │   ├── IMDBMulti
    │   └── linux
    ├── result
    │   ├── aids700nef
    │   ├── imdbmulti
    │   └── linux
    └── save
        ├── dist_mat
        ├── aids700nef_ged_astar_gidpair_dist_map.pickle
        ├── imdbmulti_ged_astar_gidpair_dist_map.pickle
        └── linux_ged_astar_gidpair_dist_map.pickle
```
(2) Specify some hyper-parameters for regression tasks in `src/ged_config.py`

(3) Train and test the model by running the following command:

`python ged_train.py`
