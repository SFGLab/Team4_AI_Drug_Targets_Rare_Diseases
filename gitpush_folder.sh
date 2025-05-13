#!/bin/bash

FOLDER=$1
MESSAGE=$2

if [ -z "$FOLDER" ] || [ -z "$MESSAGE" ]; then
  echo "Usage: ./gitpush_folder.sh <folder_path> \"commit message\""
  exit 1
fi

if [ ! -d "$FOLDER" ]; then
  echo "Error: Folder '$FOLDER' does not exist."
  exit 1
fi

git add "$FOLDER"
git commit -m "$MESSAGE"
git push origin main