import os
import jsonlines
import pandas as pd
import sys
import shutil

# Actual hateful memes format
# {"id": "94180", "img": "img/94180.png", "label": 1, "text": "happy pride month let's go beat up lesbians"}

# MMF-master
def format_mmf_master_dataset(
    item,
    parent_img_folder,
    sub_folder="mmf_master",
):
    """
    Format:
    {'id': 'covid_memes_5425',
    'image': 'covid_memes_5425.png',
    'labels': ['not harmful'],
    'text': 'gwen\n'
            '@gwenervi\n'
            'dis gon be trump tomorrow after they inject him with an\n'
            'experimental covid vaccine\n'
            '12:47 AM - Oct 2, 2020 · Twitter for iPhone\n'}
    """
    # Id needs to be able to be casted to int
    id_ = item["id"].split("_")[2]
    label_ = item["labels"][0]
    text_ = item["text"]

    # Add suffix of 0001 to prevent conflicting ids with original
    # hateful_memes dataset
    id_ = f"{id_}0001"
    img_ = f"{parent_img_folder}/{sub_folder}/{id_}.png"
    img_ = f"{parent_img_folder}/{id_}.png"
    label_ = 0 if label_ == "not harmful" else 1

    return {"id": id_, "img": img_, "label": label_, "text": text_}, item["image"]


# Multioff
def format_multioff_dataset(
    row,
    parent_img_folder,
    idx,
    sub_folder="multioff",
):
    id_ = idx
    img_ = row.image_name
    label_ = row.label
    text_ = row.sentence

    id_ = str(id_)
    img_ = f"{parent_img_folder}/{id_}.png"
    label_ = 1 if label_ == "offensive" else 0

    return {"id": id_, "img": img_, "label": label_, "text": text_}, row.image_name


def main():

    mmf_master_images_path = sys.argv[1]
    multioff_images_path = sys.argv[2]

    assert os.path.exists(
        mmf_master_images_path
    ), "MMF Master Images folder does not exist"
    assert os.path.exists(multioff_images_path), "Multioff Images folder does not exist"

    mmf_master_renamed_path = "./mmf_master"
    multioff_renamed_path = "./multioff"

    if not os.path.exists(mmf_master_renamed_path):
        os.mkdir(mmf_master_renamed_path)

    if not os.path.exists(multioff_renamed_path):
        os.mkdir(multioff_renamed_path)

    # MMF Master
    mmf_master_transformed_data = []

    mmf_master_files = [
        "./mmf_master_annotation_files/test.jsonl",
        "./mmf_master_annotation_files/train.jsonl",
        "./mmf_master_annotation_files/val.jsonl",
    ]

    for mmf_file in mmf_master_files:
        with jsonlines.open(mmf_file) as reader:
            for obj in reader:
                try:
                    formatted_obj, original_filename = format_mmf_master_dataset(
                        item=obj,
                        parent_img_folder="img",
                    )
                    mmf_master_transformed_data.append(formatted_obj)

                    original_image_path = os.path.join(
                        mmf_master_images_path, original_filename
                    )

                    new_image_path = os.path.join(
                        mmf_master_renamed_path, f"""{formatted_obj["id"]}.png"""
                    )

                    shutil.copy(original_image_path, new_image_path)

                except Exception as e:
                    print(e)

    # Multioff
    multioff_transformed_data = []

    multioff_filenames = [
        "./multioff_annotation_files/Testing_meme_dataset.csv",
        "./multioff_annotation_files/Training_meme_dataset.csv",
        "./multioff_annotation_files/Validation_meme_dataset.csv",
    ]

    file_idx = 500002

    for filename in multioff_filenames:

        df = pd.read_csv(filename)

        for idx, row in df.iterrows():
            try:
                formatted_row, original_filename = format_multioff_dataset(
                    row=row,
                    idx=file_idx,
                    parent_img_folder="img",
                )
                multioff_transformed_data.append(formatted_row)

                original_image_path = os.path.join(
                    multioff_images_path, original_filename
                )

                new_image_path = os.path.join(
                    multioff_renamed_path, f"""{formatted_row["id"]}.png"""
                )

                shutil.copy(original_image_path, new_image_path)

            except Exception as e:
                print(e)

            file_idx += 1

    # Original train, dev_seen, dev_unseen
    original_data = []

    # There are overlaps between the three
    existing_original_ids = []

    original_dev_unseen_filename = (
        "./original_hateful_memes_annotation_files/dev_unseen.jsonl"
    )

    dev_unseen_ids = []

    with jsonlines.open(original_dev_unseen_filename) as reader:
        for obj in reader:
            dev_unseen_ids.append(obj["id"])

    original_annotation_files = [
        "./original_hateful_memes_annotation_files/train.jsonl",
        "./original_hateful_memes_annotation_files/dev_seen.jsonl",
    ]

    for filename in original_annotation_files:
        with jsonlines.open(filename) as reader:
            for obj in reader:
                if (
                    obj["id"] not in existing_original_ids
                    and obj["id"] not in dev_unseen_ids
                ):
                    original_data.append(obj)
                    existing_original_ids.append(obj["id"])

    # Write to new .jsonl file
    with jsonlines.open("mmf_train_expanded.jsonl", mode="w") as writer:
        [writer.write(obj) for obj in original_data]
        [writer.write(obj) for obj in mmf_master_transformed_data]
        [writer.write(obj) for obj in multioff_transformed_data]

    with jsonlines.open("mmf_train_expanded.jsonl") as reader:
        ids = [obj["id"] for obj in reader]

    assert len(ids) == len(
        set(ids)
    ), "There are duplicate IDs detected in the new .jsonl file"


if __name__ == "__main__":
    main()
