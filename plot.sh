#!/bin/bash

# Check for exactly 2 arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 ARG1 ARG3"
  exit 1
fi

ARG1=$1
ARG3=$2

# Validate both are integers
if ! [[ "$ARG1" =~ ^-?[0-9]+$ && "$ARG3" =~ ^-?[0-9]+$ ]]; then
  echo "Error: Both arguments must be integers."
  exit 2
fi

ARG2=$((ARG1 + 1))
FILENAME="enet_inner_layer_0_${ARG1}_${ARG2}_${ARG3}_1.csv"

# Download the file via scp
echo "Downloading $FILENAME..."
scp cares-pc@128.175.213.244:~/Downloads/"$FILENAME" ~/Documents/fault_injection_research

LOCAL_PATH=~/Documents/"$FILENAME"
if [ ! -f "$LOCAL_PATH" ]; then
  echo "Error: File not found at $LOCAL_PATH"
  exit 3
fi

echo "Running 3D plot..."
cd ~/Documents
python3 3d-plot.py "$FILENAME" faulty
