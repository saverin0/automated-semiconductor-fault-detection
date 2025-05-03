import pandas as pd
import os
import re
import shutil

training_batch_file_directory = '/workspaces/automated-semiconductor-fault-detection/Training_Batch_Files'
file_name = os.listdir(training_batch_file_directory)
good_file_path = '/workspaces/automated-semiconductor-fault-detection/Validated_Batch_Files'
bad_file_path = '/workspaces/automated-semiconductor-fault-detection/Bad_Batch_Files'

pattern = r"^wafer_\d{8}_\d{6}.csv"
regex = re.compile(pattern)

if not os.path.exists(good_file_path):
    os.makedirs(good_file_path)
    print('new good files folder created', good_file_path)
else:
    print('folder already present')

if not os.path.exists(bad_file_path):
    os.makedirs(bad_file_path)
    print('new bad files folder created',bad_file_path)
    
for i in file_name:
    if re.match(regex,i):
        source_path = os.path.join(training_batch_file_directory,i)
        destination_path = os.path.join(good_file_path,i)
        shutil.copy2(source_path,destination_path)
        print(f'Files copied {i}')
    else:
        source_path = os.path.join(training_batch_file_directory,i)
        destination_path = os.path.join(bad_file_path,i)
        shutil.copy2(source_path,destination_path)
        print(f'Files copied {i}')