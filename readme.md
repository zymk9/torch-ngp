# Instance Field Training



This repository is the official PyTorch implementation of **Instance Field Training** of [Instance-NeRF](https://github.com/lyclyc52/Instance_NeRF). It is built based on [torch-ngp](https://github.com/ashawkey/torch-ngp). For the  technical description, please refer to Section 3.1 of our paper.



## Install

You can also refer to torch-ngp instructions [here](https://github.com/ashawkey/torch-ngp#install) to set up the environment. 




## Usage

#### Dataset Orginzation

We follow the data structure of torch-ngp, which uses a json file to store the camera parameters and a folder to store masks or images.

```

3dfront_sample
  |- images/masks
  |  └-...
  └- transforms.json       

```



### NeRF Training

You can run the shell script `train_3dfront.sh` to train a NeRF model.

```bash
bash train_3dfront.sh
```
If you would like to train multiple scenes, you can refer to [batch_train.py](./batch_train.py). This python script would help in the following sections related to NeRF training.



### Instance Field Training

After you obtain the NeRF model, you can run the shell script `train_3dfront_mask.sh` to train an instacne field.

```bash
bash train_3dfront_mask
```



### Mask Refinement

In order to increase the accuracy of the instance masks of if you are not satisfied with the current results, you can refine the rendered masks and then use it to supervise the NeRF training.

1. First, you can render 2D masks of all training views from the instance field using `render_mask.sh`

   ```bash
   bash render_mask.sh
   ```
   You can check the rendered masks under folder `workspace/3dfront_sample/validation`
   

2. Next, you can use some existing methods to refine the rendered masks. In our paper, we adopt [CascadaPSP](https://github.com/hkchengrex/CascadePSP) to refine the rendered masks. We provide some sample codes [here](./mask_refinement/CascadaPSP_refine.py). 
3. Finally, following the last section,  you can use the refined masks to train or finetune the NeRF model



​	





## Citation
If you find Instance-NeRF useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@inproceedings{instancenerf,
    title = {Instance Neural Radiacne Field},
    author = {Liu, Yichen and Hu, Benran and Huang, Junkai and Tai, Yu-Wing and Tang, Chi-Keung},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year = {2023}
}
```

## Acknowledgement

Credits for the amazing repo [torch-ngp](https://github.com/ashawkey/torch-ngp):

```
@misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}

@article{tang2022compressible,
    title = {Compressible-composable NeRF via Rank-residual Decomposition},
    author = {Tang, Jiaxiang and Chen, Xiaokang and Wang, Jingbo and Zeng, Gang},
    journal = {arXiv preprint arXiv:2205.14870},
    year = {2022}
}
```

