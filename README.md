# MIX
ICDM19 - Anomaly Detection / Outlier Detection for Mixed data

This is the source code of the paper named **"MIX: A joint learning framework for detecting outliers in both
clustered and scattered outliers in mixed-type data"** and published in ICDM19.

## Citation
Please cite our paper if you find this code is useful.  
Xu, H., Wang, Y., Wang, Y., & Wu, Z. (2019, November). MIX: A Joint Learning Framework for Detecting Both Clustered and Scattered Outliers in Mixed-Type Data. In 2019 IEEE International Conference on Data Mining (ICDM) (pp. 1408-1413). IEEE.  
  
or in bib format:
```
@inproceedings{xu2019mix,
  title={MIX: A Joint Learning Framework for Detecting Both Clustered and Scattered Outliers in Mixed-Type Data},  
  author={Xu, Hongzuo and Wang, Yijie and Wang, Yongjun and Wu, Zhiyue},  
  booktitle={2019 IEEE International Conference on Data Mining (ICDM)},  
  pages={1408--1413},  
  year={2019},  
  organization={IEEE}  
}  
```  

## Usage
1. run main.py for sample usage.  
2. Data set format: You may want to find the sample input data set in the "data" folder. The name of categorical attributes should be named as "A1", "A2", ..., and the numerical ones are "B1", "B2", ...  
3. The input path can be an individual data set or just a folder.  
4. The performance might have slight differences between two independent runs. In our paper, we report the average auc with std over 10 runs. 


## Dependencies
```
Python 3.6
Tensorflow == 1.12.0
pandas ==0.23.0
scikit-learn == 0.19.1
numpy == 1.14.3
```

## Parameters
k and epsilon are parameters, k = 0.3 and epsilon=0.01 are recommended, but for large data sets, please consider increase epsilon, say 0.05.
