

<div align="center">


### PyTorch implementation for 

  ## Semantic Invariant Multi-view Clustering with Fully Incomplete Information

TPAMI 2023
  
[[Paper]](https://arxiv.org/abs/2305.12743)                [[Discussion]](https://github.com/PengxinZeng/2023-TPAMI-SMILE/issues)  [[More Information]](https://github.com/PengxinZeng?tab=repositories)
</div>


## Introduction

### SMILE framework
<img src="https://github.com/PengxinZeng/2023-TPAMI-SMILE/blob/main/Fig2.png"  width="740"  />

## Requirements

- Python 3.10
- PyTorch 1.12.1
- [faiss](https://anaconda.org/pytorch/faiss-gpu)
```
conda install -c pytorch faiss-gpu
```
  
## Training

Modify the ```./Utils/PathPresettingOperator.get_dataset_path```, then train the model(s):
```train
# NoisyMNIST 
python main.py --dataset NoisyMNIST30000 --seed 9116  --aligned_prop 1 --complete_prop 1
  
# MNISTUSPS 
python main.py --dataset MNISTUSPS --seed 9116  --aligned_prop 1 --complete_prop 1
  
# Caltech     
python main.py --dataset 2view-caltech101-8677sample --seed 9116    --aligned_prop 1 --complete_prop 1
  
# CUB 
python main.py --dataset cub_googlenet_doc2vec_c10 --seed 9116    --aligned_prop 1 --complete_prop 1

# YouTubeFaces    
python main.py --dataset YouTubeFaces --seed 9116  --aligned_prop 1 --complete_prop 1
```

## Model Zoo
The pre-trained models are available [here](https://drive.google.com/drive/folders/1M1XM5-ceeW6paYf4WaPFwfbcs_6wOg8_?usp=sharing). 


Download the models, then:
```
python main.py --dataset dataset --seed seed --resume PathToYourModel
```

## Experiment Results:
<img src="https://github.com/PengxinZeng/2023-TPAMI-SMILE/blob/main/Exp2.png"  width="740"  />
<img src="https://github.com/PengxinZeng/2023-TPAMI-SMILE/blob/main/Exp3.png"  width="600"  />



## Citation

If SMILE is useful for your research, please cite the following paper:

```
@article{zeng2023semantic,
  title={Semantic Invariant Multi-view Clustering with Fully Incomplete Information},
  author={Zeng, Pengxin and Yang, Mouxing and Lu, Yiding and Zhang, Changqing and Hu, Peng and Peng, Xi},
  journal={arXiv preprint arXiv:2305.12743},
  year={2023}
}
```





































