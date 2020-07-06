# HGMN: Hierarchical Graph Matching Networks for Deep Graph Similarity Learning


# Run the codes

## 1. Requirements
Make sure you have installed all of packages or libraries (including dependencies if necessary) mentioned in `requirements.txt` on your machine.


## 2. Usages

### 2.1 Classification task
(1) Prepare the dataset for the classification task.

Datasets for the classification task is provided in `data/CFG` directory.

(2) Specify some hyper-parameters for classification tasks in `src/cfg_config.py`

(3) Train and test the model by running the following command:

```
cd src
python cfg_train.py
```

### 2.2 regression task
(1) Prepare the dataset for regression tasks

Data for regression task are placed in `/data/GED` directory. All the files required by our codes can be downloaded following instructions in this [repo](https://github.com/yunshengb/SimGNN).
Please make sure you have downloaded all the 3 directories required by our code: `data`, `save`, and `result`.

After downloading these files, please put them under `/data/GED`, which is the default data folder by our configuration, or you can also specify your own data directory.

An example directory structure is:
``` 
data
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

```
cd src
python ged_train.py
```
