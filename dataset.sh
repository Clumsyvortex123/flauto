#!/bin/bash
# Get the current working directory 
path = $(pwd)
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
cd $path

