# Initializing Environment

In order to run all the experiments, several parts will need to be installed to
cater to the various models used as image/text encoders. Start with a blank
conda environment:

```
conda  create -n cs7643_project python=3.7
conda activate cs7643_project
```

## MMF

The experiments are done via Facebook's MMF [framework](https://github.com/facebookresearch/mmf).
The framework can be used with a custom directory, but will need to be first cloned to a separate
folder and installed separately. Official installation guides can be found [here](https://mmf.sh/docs/).


```
# In a separate folder
git clone https://github.com/facebookresearch/mmf.git

cd mmf

# This will install mmf in the created conda environment
pip install --editable .
```

Pytorch will be installed according to MMF's requirements, and depending on the CUDA version
needed, you might need to overwrite the installation version that still conforms to the
MMF's version [requirement](https://github.com/facebookresearch/mmf/blob/6f3f40f56c6a7f5132235aa117f61d1ba693223e/requirements.txt).
Details on installing different Pytorch versions with specific CUDA versions can be found [here](https://pytorch.org/get-started/previous-versions/).


## EfficientNet

Efficient Net needs to be installed separately with `--no-deps` as else it will break MMF pytorch requirements.

```
pip install --no-deps efficientnet_pytorch==0.7.1
```

## Other Requirements

Other requirements (for [Yolov5](https://github.com/ultralytics/yolov5) and HuggingFace) are
wrapped in `requirements.txt`

```
pip install -r requirements.txt
```

# Building the Hateful Memes Dataset

To build the Hateful Memes dataset, `mmf` will need to be installed first. 


# Running The Experiments

### BERT
```
mmf_run config=mmf/projects/hateful_memes/configs/unimodal/bert.yaml     model=unimodal_text     dataset=hateful_memes
```

### RoBERTa
Modify the `UnimodalBase` class in `mmf/mmf/models/unimodal.py`
```
mmf_run config=mmf/projects/hateful_memes/configs/unimodal/roberta.yaml     model=unimodal_text     dataset=hateful_memes
```

### BERT finetuned on hateful / harmful meme texts
```
mmf_run config=mmf/projects/hateful_memes/configs/unimodal/bert_hateful.yaml     model=unimodal_text     dataset=hateful_memes
```

### Visual BERT
```
mmf_run config=mmf/projects/hateful_memes/configs/visual_bert/direct.yaml     model=visual_bert     dataset=hateful_memes
```

### Visual BERT with BERT finetuned on hateful / harmful meme texts
```
mmf_run config=mmf/projects/hateful_memes/configs/visual_bert/visual_bert_hateful.yaml     model=visual_bert     dataset=hateful_memes
```

# Expanding the Hateful Memes Dataset


# Running The Experiments

## Unimodal

```
# Yolov5
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/yolov5s/defaults.yaml model=unimodal_yolov5s dataset=hateful_memes
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/yolov5m/defaults.yaml model=unimodal_yolov5m dataset=hateful_memes
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/yolov5l/defaults.yaml model=unimodal_yolov5l dataset=hateful_memes
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/yolov5x/defaults.yaml model=unimodal_yolov5x dataset=hateful_memes

# MobileNet
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/mobilenetv3_large/defaults.yaml model=unimodal_mobilenetv3_large dataset=hateful_memes

# Torchvision Encoders
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/torchvision-encoders/resnext50_32x4d.yaml model=unimodal_image dataset=hateful_memes
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/torchvision-encoders/resnext101_32x8d.yaml model=unimodal_image dataset=hateful_memes
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/torchvision-encoders/wide_resnet50_2.yaml model=unimodal_image dataset=hateful_memes
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/torchvision-encoders/wide_resnet101_2.yaml model=unimodal_image dataset=hateful_memes

# BERT
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/bert/defaults.yaml model=unimodal_text dataset=hateful_memes
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/bert/finetuned_on_hateful_memes.yaml model=unimodal_text dataset=hateful_memes

# RoBERTa -- CUBLAS issue, others to test
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/roberta/defaults.yaml model=unimodal_text_modified_roberta datset=hateful_memes
```

## Multimodal

```
# Visual BERT
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/visual_bert/direct.yaml model=visual_bert dataset=hateful_memes
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/visual_bert/finetuned_on_hateful_mememes.yaml model=visual_bert dataset=hateful_memes
```
