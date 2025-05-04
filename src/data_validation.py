import pandas as pd
import os
import re
import shutil
import csv

training_batch_file_directory = '/workspaces/automated-semiconductor-fault-detection/Training_Batch_Files'
file_name = os.listdir(training_batch_file_directory)
good_file_path = '/workspaces/automated-semiconductor-fault-detection/Validated_Batch_Files'
bad_file_path = '/workspaces/automated-semiconductor-fault-detection/Bad_Batch_Files'
training_column_count = 592
prediction_column_count = 591

pattern = r"^wafer_\d{8}_\d{6}.csv"
regex = re.compile(pattern)

if os.path.exists(good_file_path):
    os.removedirs(good_file_path)
    print('Previous run folder got deleted')
    os.makedirs(good_file_path)
    print('new good files folder created', good_file_path)

if not os.path.exists(bad_file_path):
    os.makedirs(bad_file_path)
    print('new bad files folder created',bad_file_path)
    
for i in file_name:
    if re.match(regex,i):
        with open(os.path.join(training_batch_file_directory,i),'r',encoding='utf-8') as file:
            reader = csv.reader(file)
            first_row = next(reader)
            if training_column_count == len(first_row):
                source_path = os.path.join(training_batch_file_directory,i)
                destination_path = os.path.join(good_file_path,i)
                shutil.copy2(source_path,destination_path)
                print(f'Files copied {i}')
            else:
                source_path = os.path.join(training_batch_file_directory,i)
                destination_path = os.path.join(bad_file_path,i)
                shutil.copy2(source_path,destination_path)
                print(f'Correct Name but missing columns Files copied {i}')
    else:
        source_path = os.path.join(training_batch_file_directory,i)
        destination_path = os.path.join(bad_file_path,i)
        shutil.copy2(source_path,destination_path)
        print(f'Bad File copied {i}')