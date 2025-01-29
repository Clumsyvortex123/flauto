#!/bin/bash
# Create a new directory (you can change the folder name)
NEW_DIR=~/Bimetal

# Check if the directory already exists
if [ ! -d "$NEW_DIR" ]; then
    echo "Creating new directory: $NEW_DIR"
    mkdir "$NEW_DIR"
else
    echo "Directory $NEW_DIR already exists."
fi

# Move all files from the current repository to the new directory
echo "Moving files into $NEW_DIR..."
cd ..
mv flauto "$NEW_DIR/"
mv .[^.]* "$NEW_DIR/"  # This moves hidden files/folders like .gitignore, .git, etc.

# Check if files were moved successfully
if [ $? -eq 0 ]; then
    echo "Files successfully moved to $NEW_DIR."
else
    echo "Error moving files."
fi
cd $NEW_DIR
# Set the virtual environment name
VENV_NAME="venv"

# Step 1: Update the system and install required tools
echo "Updating system and installing required tools..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip zellij ranger lazygit htop nano wget curl

# Install Visual Studio Code
echo "Installing Visual Studio Code..."
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /usr/share/keyrings/
sudo sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install -y code
rm -f packages.microsoft.gpg

# Step 2: Create the virtual environment
echo "Creating a virtual environment: $VENV_NAME..."
python3 -m venv $VENV_NAME

# Step 3: Activate the virtual environment
echo "Activating the virtual environment..."
source $VENV_NAME/bin/activate

# Step 4: Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Step 5: Install required Python packages
echo "Installing required Python packages..."
pip install tensorflow opencv-python numpy nicegui matplotlib pandas scipy scikit-learn pillow  tensorflow-datasets torch torchvision torchaudio tensorboard plotly seaborn ultralytics  

# Step 6: Confirm installation
echo "Installation complete!"
echo "The virtual environment '$VENV_NAME' is ready to use."
echo "To activate it manually in the future, run:"
echo "source $VENV_NAME/bin/activate"
cd flauto
