#!/bin/bash

input_dir="$1"
output_dir="$2"

find "$input_dir" -type f -print0 | while IFS= read -r -d $'\0' file
do
    name=$(basename "$file")
    output_file="$output_dir/$name"
    python distance_transform.py \
        --input "$file" \
        --output "$output_file"
done
