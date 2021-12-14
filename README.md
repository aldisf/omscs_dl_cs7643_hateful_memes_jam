# Initializing Environment

In order to run all the experiments, several parts will need to be installed to
cater to the various models used as image/text encoders. Start with a blank
conda environment:

```
conda create -n cs7643_project python=3.7
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

To build the Hateful Memes dataset, `mmf` will need to be installed first. After `mmf` has been installed,
the Hateful Memes dataset can then be built by following these steps:

1. Download the Hateful Memes dataset from https://hatefulmemeschallenge.com/ (~4GB)
2. Unzip the Hateful Memes dataset: 

```
unzip hateful_memes.zip
```

3. To use the MMF framework, convert the dataset to MMF format: 

```
mmf_convert_hm --zip_file hateful_memes.zip --password '' --bypass_checksum=1
```

4. Modify the directory structure to match the following:

```
#   defaults
#       annotations
#           <*.jsonl files>
#       images
#           data
#               <metadata_txt_files>
#           img
#               <all the images>
```

The following shell commands can help to achieve step (4).

```
mkdir ~/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations

cp ~/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/hateful_memes/*.jsonl ~/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations

mkdir ~/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/

cp -r ~/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/hateful_memes/img ~/.cache/torch/

mmf/data/datasets/hateful_memes/defaults/images/
```

You may get an error saying that `train.jsonl` does not exist when running `mmf_convert_hm` in step
(2) above. Seems like it can be ignored as long as the directory structures and content are arranged
as above.

# Expanding the Hateful Memes Dataset

We expand the dataset directly by adding images to the folder and the `train.jsonl` file under
the annotations folder. The jsonlines file will specify which images are to be used as the
training set, together with their specified id, path, label, and text information.

We are using the [HarMeme](https://github.com/di-dimitrov/mmf) and [MultiOFF](https://drive.google.com/drive/folders/1hKLOtpVmF45IoBmJPwojgq6XraLtHmV6) datasets as expansions. To download them:

### HarMeme Dataset
1. Download the zip file of the repository https://github.com/di-dimitrov/mmf
2. Annotation files are located in `data/datasets/memes/defaults/annotations`
3. Images are located in `data/datasets/memes/defaults/images`

### MultiOFF Dataset
1. Download the dataset from Google Drive: https://drive.google.com/drive/folders/1hKLOtpVmF45IoBmJPwojgq6XraLtHmV6
2. Annotation files are located in the Split Dataset directory
3. Images are located in the Labelled Images directory

After downloading the files, you can refer to the specific instructions inside the `./dataset_expansion` folder [here](./dataset_expansion/README.md)

# Running The Experiments

### BERT
```
mmf_run config=mmf/projects/hateful_memes/configs/unimodal/bert.yaml     model=unimodal_text     dataset=hateful_memes
```

### BERT (Cased)
```
mmf_run config=mmf/projects/hateful_memes/configs/unimodal/bert_cased.yaml     model=unimodal_text     dataset=hateful_memes
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

# EfficientNet
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/efficientnet/unimodal/efficientnet-b0.yaml model=unimodal_efficientnet dataset=hateful_memes

# BERT
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/bert/defaults.yaml model=unimodal_text dataset=hateful_memes
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/bert/finetuned_on_hateful_memes.yaml model=unimodal_text dataset=hateful_memes

# RoBERTa
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/roberta/defaults.yaml model=unimodal_text_modified_roberta datset=hateful_memes
```

## Multimodal

```
# Late Fusion (EfficientNet + BERT)
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/efficientnet/late_fusion/efficientnet-b0.yaml model=late_fusion_efficientnet dataset=hateful_memes

# Concat (EfficientNet + BERT)
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/efficientnet/concat_bert/efficientnet-b0.yaml model=concat_bert_efficientnet dataset=hateful_memes

# Visual BERT
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/visual_bert/direct.yaml model=visual_bert dataset=hateful_memes
MMF_USER_DIR="$(pwd)" mmf_run config=configs/experiments/visual_bert/finetuned_on_hateful_mememes.yaml model=visual_bert dataset=hateful_memes
```


# References

Some codes are taken from Facebook's MMF repository. Those files will have comments
at the top noting so, and the `LICENSE` from Facebook MMF's Github will be present in the
folder.

https://github.com/facebookresearch/mmf