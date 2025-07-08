#!/bin/bash

if [ -z "$2" ]; then
    echo "Usage: $0 <cmd_filename> <output_filename>"
    exit 1
fi

input_file="$1"
output_file="$2"

prefix_lines=$(sed '/OC/ q' "$input_file")
prefix_output=$(echo "$prefix_lines" | bin/processor_tool_risp)
length=$(echo "$prefix_output" | wc -l)

all_output=$(cat "$input_file" | bin/processor_tool_risp)
IFS=$'\n' read -d '' -r -a output_lines <<< "$all_output"

> "$output_file"
count=0
for line in "${output_lines[@]}"; do
    echo "$line" >> "$output_file"
    ((count++))
    if (( count % length == 0 )); then
        echo "" >> "$output_file"
    fi
done
