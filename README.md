# CCD
Official PyTorch implementation of the paper ["Contextual Debiasing for Visual Recognition with Causal Mechanisms"](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Contextual_Debiasing_for_Visual_Recognition_With_Causal_Mechanisms_CVPR_2022_paper.pdf)

A Simple but Effective Baseline for Multi-label Classification

## Requirements:
- PyTorch >= 1.7.0
- torchvision >= 0.5.0
- randaugment
- pycocotools
- opencv-python
- pillow
## Training
To train a model, run train.py with the desired model architecture and the path to the dataset. For example, to train with full model on MS-COCO with resnet101 backbone:
```
CUDA_VISIBLE_DEVICES=0,1 python3 -u train.py --batch-size=128 --image-size=448 --lr=1e-4 --backbone=resnet101  --loss=focal --use_intervention=True --use_tde=True --stop_epoch=5
```
to train with full model on MS-COCO with Swin-B backbone:
```
CUDA_VISIBLE_DEVICES=0,1 python3 -u train.py --batch-size=32 --image-size=384 --lr=5e-5 --backbone=swim_transformer --loss=halfasl --use_tde=True --use_intervention=True --stop_epoch=2
```
# Citation
If you find our paper or this project helps your research, please kindly consider citing our paper in your publications.
```bash
@inproceedings{liu2022contextual,
  title={Contextual Debiasing for Visual Recognition With Causal Mechanisms},
  author={Liu, Ruyang and Liu, Hao and Li, Ge and Hou, Haodi and Yu, TingHao and Yang, Tao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12755--12765},
  year={2022}
}
```
