## Usage

To use this dataset expansion script, you will need to complete
the initial hateful memes dataset building step first, and have
downloaded the HarMeme and Multioff datasets.

You will need the paths to the images folder of both
HarMeme and MultiOff dataset in your machine, to use the script, run:

```
python expand_hateful_memes_dataset.py PATH_TO_MFMF_MASTER_IMAGES_FOLDER PATH_TO_MULTIOFF_IMAGES_FOLDER
```

The script will generate three things:

1. Folder `./mmf_master` containing the renamed images of HarMeme dataset
2. Folder `./multioff` containing the renamed images of MultiOff dataset
3. `mmf_trained_expanded.jsonl` files containing the annotations to be used for training.

After running the script, you will need to paste the images under the generated
`./mmf_master` and `./multioff` folders in the MMF hateful_memes images folder
following the folder structure below:

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
        ├── (Paste images from the renamed HarMeme dataset here)
        └── (Paste images from Multioff dataset here)
```

After pasting the images, under the `annotations` folder you will find the original `train.jsonl`
file. Rename this file to keep it as reference or to run the original, un-expanded dataset.
You should rename the script output `mmf_train_expanded.jsonl` to `train.jsonl` and paste
it inside the annotations folder of the mmf Hateful memes dataset folder. The `train.jsonl` file
will govern which images to use for training.

If you want to run using the original dataset, change back the `train.jsonl` to the original file
that was renamed above.
