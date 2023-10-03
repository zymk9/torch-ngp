# Instance Field Training

This repository contains the NeRF training code of [Instance-NeRF](https://github.com/lyclyc52/Instance_NeRF). It is built based on [torch-ngp](https://github.com/ashawkey/torch-ngp). For the technical description, please refer to Section 3.1 of our paper.



## Install

Refer to torch-ngp instructions [here](https://github.com/ashawkey/torch-ngp#install) to set up the environment. 




## Usage

### Dataset Orginzation

We follow the data structure of torch-ngp, which uses a json file to store the camera parameters and a folder to store masks or images.

```

3dfront_sample
  |- images/masks
  |  └-...
  └- transforms.json       

```


### NeRF Training
Before training the instance field, youd need to first train a regular NeRF of the scene:

```bash
python3 main_nerf_mask.py \
${train_dir} \
--workspace ${workspace_dir} \
--iters 30000 \
--lr 1e-2 \
--bound 8 \
--gpu ${gpu_id} \
-O \
--label_regularization_weight 0.0 '
```

`${train_dir}` should be the directory containing the scene images and `transforms.json`. `${workspace_dir}` is the folder to hold the output NeRF weights and results.


### Mask Preparation
First, ensure you have obtained the discrete 3D masks of the instances by running inference with the trained NeRF-RCNN model. If you haven't, check [here]() for details.

Then, you should get the initial 2D segmentation masks using Mask2Former. Please refer to the instructions [here](../Mask2Former/README.md).


### Instance Field Training
After you obtain the regular NeRF model and the aligned masks, you can train an instacne field with the following script:

```bash
python3 main_nerf_mask.py \
${train_dir} \
--workspace ${workspace_dir} \
--iters 30000 \
--lr 1e-2 \
--bound 8 \
--gpu ${gpu_id} \
-O \
--label_regularization_weight 1.0 \
--ckpt ${checkpoint} \
--load_model_only \
--train_mask \
--num_rays 4096 \
--patch_size 8
```

`${checkpoint}` should points to the trained regular NeRF model in the previous step.


### Instance Field Inference
To render the instance field, simply append `--test` to the script above.


### Mask Refinement

In order to increase the accuracy of the instance masks of if you are not satisfied with the current results, you can refine the rendered masks and then use it to supervise the NeRF training.

1. First, you can render 2D masks of all training views from the instance field using `render_mask.sh`

```bash
bash render_mask.sh
```
You can check the rendered masks under folder `workspace/3dfront_sample/validation`
   
2. Next, you can use some existing methods to refine the rendered masks. In our paper, we adopt [CascadaPSP](https://github.com/hkchengrex/CascadePSP) to refine the rendered masks. We provide some sample codes [here](./mask_refinement/CascadaPSP_refine.py). 

3. Finally, following the last section,  you can use the refined masks to train or finetune the NeRF model



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

