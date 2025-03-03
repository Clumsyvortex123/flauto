#!/bin/bash
# Get the current working directory 
current_path=$(pwd)
# To download the dataset, create a directory
mkdir dataset

# Inside this directory, create two different directories for augmented and raw dataset

cd dataset
mkdir augmented_dataset raw_dataset

# Download the corresponding dataset from roboflow workspace using the CLI:
echo "Downloading augmented dataset..."
cd augmented_dataset 
curl -L "https://app.roboflow.com/ds/5qvXa12UsZ?key=A6fPsNVoJE" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip README.dataset.txt README.roboflow.txt
echo "Download successful."

echo "Downloading raw dataset..."
cd ..
cd raw_dataset
curl -L "https://app.roboflow.com/ds/6f6byjbJXT?key=eNVj9im3hS" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip README.dataset.txt README.roboflow.txt
echo "Download successful."

# Changing the path of the dataset.yaml files for ease of training model
# Paths for the augmented dataset
AUG_TRAIN_PATH="$current_path/dataset/augmented_dataset/train/images"
AUG_VAL_PATH="$current_path/dataset/augmented_dataset/valid/images"
AUG_TEST_PATH="$current_path/dataset/augmented_dataset/test/images"

# Paths for the raw dataset
RAW_TRAIN_PATH="$current_path/dataset/raw_dataset/train/images"
RAW_VAL_PATH="$current_path/dataset/raw_dataset/valid/images"
RAW_TEST_PATH="$current_path/dataset/raw_dataset/test/images"

AUG_YAML_FILE="$current_path/dataset/augmented_dataset/data.yaml"
RAW_YAML_FILE="$current_path/dataset/raw_dataset/data.yaml"

# Check if YAML file exists
if [ -f "$AUG_YAML_FILE" ]; then
    echo "Updating dataset paths in $AUG_YAML_FILE..."
    sed -i "s|train: .*|train: $AUG_TRAIN_PATH|" "$AUG_YAML_FILE"
    sed -i "s|val: .*|val: $AUG_VAL_PATH|" "$AUG_YAML_FILE"
    sed -i "s|test: .*|test: $AUG_TEST_PATH|" "$AUG_YAML_FILE"
    echo "YAML file updated successfully for augmented dataset."
else
    echo "Error: YAML file not found at $AUG_YAML_FILE"
fi

if [ -f "$RAW_YAML_FILE" ]; then
    echo "Updating dataset paths in $RAW_YAML_FILE..."
    sed -i "s|train: .*|train: $RAW_TRAIN_PATH|" "$RAW_YAML_FILE"
    sed -i "s|val: .*|val: $RAW_VAL_PATH|" "$RAW_YAML_FILE"
    sed -i "s|test: .*|test: $RAW_TEST_PATH|" "$RAW_YAML_FILE"
    echo "YAML file updated successfully for raw dataset."
else
    echo "Error: YAML file not found at $RAW_YAML_FILE"
fi

cd $path
