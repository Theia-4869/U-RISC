# EM cytomembrane segmentation with neural network ensemble and morphological processing

## Environment
#### Hardware

- 4 NVIDIA 3090Ti GPUs (24GB memory each)
- 32 CPUs

#### Packages
```bash
pip install -r requirements.txt
```

## Data
Processed data and pretrained ResNet (50 and 152) can be downloaded [here](https://pan.baidu.com/s/1LrP56-fstinTh3cNUtTRKg). Put it in top-level folder.

## Model
#### Simple Track
[DFF](https://arxiv.org/abs/1902.09104), backbone ResNet-50

#### Complex Track
[CASENet](https://arxiv.org/abs/1705.09759), backbone ResNet-152


## Training
#### Simple Track
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset simple --model DFF --backbone resnet50 --batch-size 8 --lr 0.001 --epochs 200 --crop-size 960 --kernel-size 5 --edge-weight 0.4
```
#### Simple Track (Visualization)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_vis.py --dataset simple --model DFF --backbone resnet50 --batch-size 8 --lr 0.001 --epochs 200 --crop-size 960 --kernel-size 5 --edge-weight 0.4
```
#### Complex Track
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset complex --model CASENet --backbone resnet152 --batch-size 4 --lr 0.001 --epochs 100 --crop-size 1280 --kernel-size 9 --edge-weight 0.4
```
#### Complex Track (Visualization)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_vis.py --dataset complex --model CASENet --backbone resnet152 --batch-size 4 --lr 0.001 --epochs 100 --crop-size 1280 --kernel-size 9 --edge-weight 0.4
```

## Validaing and Morphological Processing
#### Simple Track
```bash
CUDA_VISIBLE_DEVICES=0 python val.py --dataset simple --model DFF --backbone resnet50
```
#### Complex Track
```bash
CUDA_VISIBLE_DEVICES=0 python val.py --dataset complex --model CASENet --backbone resnet152
```
#### Morphological Processing
```bash
python val_mor.py
```

## Testing and Ensembling
#### Simple Track
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset simple --model DFF --backbone resnet50
```
#### Complex Track
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset complex --model CASENet --backbone resnet152
```
