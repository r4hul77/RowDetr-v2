import os
import json
import argparse
import csv
import pandas as pd
def process_directories(parent_dir, output_file):
    # Prepare the CSV file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['dir-name', 'PolyOptLoss_val', 'Mean_LPD', 'Total_LPD', 'TuSimple_FNR', 'TuSimple_FPR', 'TuSimple_F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        df = pd.DataFrame()
        # Iterate through each subdirectory
        for dirpath, dirnames, filenames in os.walk(parent_dir):
            for dirname in dirnames:
                current_dir = os.path.join(dirpath, dirname)

                # Look for a JSON file in the current directory
                json_file = next((f for f in os.listdir(current_dir) if f.endswith('.json')), None)
                if json_file:
                    json_path = os.path.join(current_dir, json_file)

                    # Extract the parent directory (e.g., "3-2")
                    parent_dir_name = os.path.basename(os.path.dirname(current_dir))

                    # Read and parse the JSON file
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # Extract required fields
                    row = {
                        'dir-name': parent_dir_name,
                        'PolyOptLoss_val': data.get('PolyOptLoss_val', None),
                        'Mean_LPD': data.get('Mean LPD', None),
                        'Total_LPD': data.get('Total LPD', None),
                        'TuSimple_FNR': data.get('TuSimple FNR', None),
                        'TuSimple_FPR': data.get('TuSimple FPR', None),
                        'TuSimple_F1': data.get('TuSimple F1', None)
                    }
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    # Write the row to the CSV file
                    writer.writerow(row)
        tab = df.to_latex()
        with open(f"{output_file.split('.')[0]}.tab", "w") as tab_file:
            print("Writing to ", f"{output_file.split('.')[0]}.tab")
            tab_file.write(tab)
                
def main():
    parser = argparse.ArgumentParser(description="Generate a metrics table from directories with JSON files.")
    parser.add_argument("parent_dir", type=str, help="Path to the parent directory containing the subdirectories.")
    parser.add_argument("output_file", type=str, help="Path to the output CSV file.")
    args = parser.parse_args()

    process_directories(args.parent_dir, args.output_file)
    print(f"Table generated and saved to {args.output_file}")

if __name__ == "__main__":
    main()
