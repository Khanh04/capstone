#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install git if not installed
if ! command_exists git; then
    echo "Git is not installed. Installing Git..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install git
    else
        sudo apt-get update && sudo apt-get install -y git
    fi
else
    echo "Git is already installed."
fi

# Install Node.js if not installed
if ! command_exists node; then
    echo "Node.js is not installed. Installing Node.js..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install node
    else
        sudo apt-get update && sudo apt-get install -y nodejs npm
    fi
else
    echo "Node.js is already installed."
fi

# Clone the repository only if SillyTavern directory does not exist
if [[ ! -d "SillyTavern" ]]; then
    echo "Cloning the SillyTavern repository (release branch)..."
    git clone https://github.com/SillyTavern/SillyTavern -b release
else
    echo "SillyTavern directory already exists. Skipping clone."
fi

# Navigate into the folder
cd SillyTavern || { echo "Failed to navigate to SillyTavern directory."; exit 1; }

# Run the start.sh script
if [[ -f "start.sh" ]]; then
    echo "Running the start.sh script..."
    chmod +x start.sh
    ./start.sh
else
    echo "start.sh script not found in the cloned repository."
    exit 1
fi
