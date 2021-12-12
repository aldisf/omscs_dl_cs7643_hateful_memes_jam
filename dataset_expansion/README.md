## Usage

```python expand_hateful_memes_dataset.py PATH_TO_MFMF_MASTER_IMAGES_FOLDER PATH_TO_MULTIOFF_IMAGES_FOLDER```

The script assumes that the new data sources will be added in a sub-directory fashion in the mmf dataset folder
instead of placing them all in one folder, to enable easy reset to the original dataset. 

Your MMF hateful_memes dataset folder should look like: 

```
.
├── annotations
│   ├── dev_seen.jsonl
│   ├── dev_unseen.jsonl
│   ├── test_seen.jsonl
│   ├── test_unseen.jsonl
│   ├── train.jsonl (rename mmf_train_expanded.jsonl to this)
│   └── train__.jsonl (original file, keep for reference)
├── extras
│   └── vocabs
│       └── vocabulary_100k.txt
├── extras.tar.gz
└── images
    ├── data
    └── img
        ├── 01235.png
        ├── 01236.png
        ├── 01243.png
        ├── (Other original MMF hateful_memes dataset imgs)
        └── mmf_master
            └── (Paste images from HarMeme dataset here)
        └── multioff
            └── (Paste images from Multioff dataset here)
```
