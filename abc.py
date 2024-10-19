import os
import json

# Function to read and process each JSON file
def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        source = data[3].replace("./raw_txt_input/", "")  # Remove 'raw_txt_input' from source
        metadata = os.path.splitext(os.path.basename(file_path))[0]  # Remove file extension from metadata
        return {
            "query": data[0],
            "response": data[1],
            "context": data[2],
            "source": source,
            "metadata": metadata
        }

# Function to read all JSON files in a directory and consolidate the data
def read_json_files_in_directory(directory):
    consolidated_data = []
    for file_name in sorted(os.listdir(directory)):  # Sort files by name
        if file_name.endswith('.json'):
            file_path = os.path.join(directory, file_name)
            consolidated_data.append(process_json_file(file_path))
    return consolidated_data

# Directory containing the JSON files
input_directory_path = './output/qatuples_raw'
if not os.path.exists(input_directory_path):
    os.makedirs(input_directory_path)

# Create the 'extracted_data' folder if it doesn't exist
output_directory_path = './extracted_data'
if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)

# Read and consolidate the data from all JSON files in the directory
consolidated_data = read_json_files_in_directory(input_directory_path)

# Write the consolidated data to a single JSON file in the 'extracted_data' folder
output_file_path = os.path.join(output_directory_path, 'consolidated_data.json')
with open(output_file_path, 'w') as f:
    json.dump(consolidated_data, f, indent=4)

print("Consolidated data has been written to:", output_file_path)