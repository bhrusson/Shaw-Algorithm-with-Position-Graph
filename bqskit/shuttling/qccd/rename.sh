#!/bin/bash

# Check if the folder path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <folder_path>"
  exit 1
fi

# Assign the folder path
folder_path="$1"

# Check if the folder exists
if [ ! -d "$folder_path" ]; then
  echo "Error: Folder '$folder_path' does not exist."
  exit 1
fi

# Loop through all files in the folder and remove ":" from their names
for file in "$folder_path"/*; do
  # Skip if it's not a file
  [ -f "$file" ] || continue

  # Get the base name and directory of the file
  base_name=$(basename "$file")
  dir_name=$(dirname "$file")

  # Replace ":" with an empty string in the file name
  new_name="${base_name//:/}"

  # Rename the file
  mv "$file" "$dir_name/$new_name" && echo "Renamed: '$file' to '$dir_name/$new_name'"
done

echo "Completed renaming files in $folder_path."
