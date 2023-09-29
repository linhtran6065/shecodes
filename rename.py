import os
import re

# Specify the folder containing the files
folder_path = 'D:\Desktop\shecodes\data\Diep_nu'

# Define a regular expression pattern to extract the number part
pattern = re.compile(r'\d+')

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a regular file (not a directory)
    if os.path.isfile(os.path.join(folder_path, filename)):
        # Extract the number part from the old name using regex
        match = pattern.search(filename)
        if match:
            number_part = match.group()
            
            # Construct the new name using the extracted number
            new_name = f'{int(number_part)+22}.m4a'  # You can customize the file extension
            
            # Rename the file
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))