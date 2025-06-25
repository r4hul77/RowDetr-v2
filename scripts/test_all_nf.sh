#!/bin/bash

# Parent directory containing all the directories
parent_dir=$1
save_dir=$2
# Loop through each subdirectory
for dir in "$parent_dir"/*; do
    if [ -d "$dir" ]; then
        # Extract the directory name
        dir_name=$(basename "$dir")

        # Get the path to the config file (.py file)
        config_file=$(find "$dir" -maxdepth 1 -name "*.py" | head -n 1)

        # Get the path to the best_F1 @ 10_epoch_*.pth file
        best_f1_file=$(find "$dir" -maxdepth 2 -name "best_F1\ @\ 10_epoch_*.pth" | head -n 1)

        # Ensure the required files exist before running the command
        if [ -n "$config_file" ] && [ -n "$best_f1_file" ]; then
            # Output directory for the test results
            output_dir="${save_dir}/${dir_name}"
            mkdir -p "$output_dir"

            # Execute the Python test command
            echo "Running: python /home/r4hul-lcl/Projects/row_detection/test_scripts/test_nf.py --config $config_file --checkpoint $best_f1_file --output $output_dir"
            python test_scripts/test_nf.py --config "$config_file" --checkpoint "$best_f1_file" --output "$output_dir"
        else
            echo "Skipping $dir_name: Required files not found"
        fi
    fi
done
python tools/generate_table.py $save_dir $save_dir/metrics.csv