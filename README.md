# SOLOV2_FSOLO
The model is best mask
## **INSTALLATION**

### **Requirements**

```bash
conda creat --name [Your envirement name] python=3.7
```
First install Detectron2 following the official guide: refer [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md):

**Install some libary.**
```bash
python -m pip install pyyaml==5.1
```
**Install cuda 11.1 + torch 1.9.**
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
**Install Detectron2 from source follwing the local clone.**
To rebuild detectron2 that's built from a local clone, use `rm -rf build/ **/*.so` to clean the old build first. You often need to rebuild detectron2 after reinstalling PyTorch.
```python
rm -rf build/ **/*.so
```
```bash
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

## Then build AdelaiDet
```bash
git clone https://github.com/aim-uofa/AdelaiDet.git
cd AdelaiDet
python setup.py build develop
```
**Some projects may require special setup, please follow their own README.md in [configs](https://github.com/aim-uofa/AdelaiDet/tree/master/configs)**

# Quick Start.

## Inference with Pre-Trained Models.
1. Pick a model and its config file, for example, fcos_R_50_1x.yaml.
2. Download the model `wget https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download -O fcos_R_50_1x.pth`.
3. Run the demo with
```bash
wget https://cloudstor.aarnet.edu.au/plus/s/chF3VKQT4RDoEqC/download -O SOLOv2_R50_3x.pth
python demo/demo.py \
    --config-file configs/SOLOv2/R50_3x.yaml \
    --input input1.jpg input2.jpg \
    --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth
```
## Train Your Own Models.

**To train a model with "train_net.py", first setup the corresponding datasets following [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md), then run:**
```bash
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/SOLOv2/R50_3x.yaml \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/SOLOv2_R50_3x
```
**To evaluate the model after training, run:**
```bash
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/SOLOv2/R50_3x.yaml \
    --eval-only \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/SOLOv2_R50_3x \
    MODEL.WEIGHTS training_dir/SOLOv2_R50_3x/model_final.pth
```
**Note That:**
* The configs are made for 8-GPU training. To train on another number of GPUs, change the `--num-gpus`.
* If you want to measure the inference time, please change `--num-gpus` to 1.
* We set `OMP_NUM_THREADS=1` by default, which achieves the best speed on our machines, please change it as needed.
* This quick start is made for FCOS. If you are using other projects, please check the projects' own `README.md` in [configs](https://github.com/aim-uofa/AdelaiDet/tree/master/configs).




