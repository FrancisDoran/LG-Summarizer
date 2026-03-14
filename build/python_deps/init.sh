#!/bin/bash

# this script makes sure a virtual environment is created and dependencies are installed

CURRENT_PATH=$(pwd)

# check if venv exists, if not then create it
if [[ ! -d "$CURRENT_PATH/venv" ]]; then
    echo "No virtual environment found. Creating a new one now."
    python3 -m venv venv
    echo "Virtual environment created. Please source with source venv/bin/activate and run this script again."
    exit 0
fi

# check to make sure the virtual environment is activated before installing dependencies
if [[ -z $VIRTUAL_ENV ]]; then
    echo "Need to source the virtual environment with: source venv/bin/activate"
    exit 1
fi

# when the previous checks are passed, then we can install dependencies
echo "Updating dependencies..."
chmod +x install_deps.sh
./install_deps.sh

cat << EOF
 
Virtual environment created and dependencies installed.

Run the code with: python -m src.main
EOF

