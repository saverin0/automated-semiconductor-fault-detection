import pandas as pd
import os
import re
import shutil
import csv

training_batch_file_directory = '/workspaces/automated-semiconductor-fault-detection/Training_Batch_Files'
training_batch_files_folder = os.listdir(training_batch_file_directory)
training_good_file_path = '/workspaces/automated-semiconductor-fault-detection/Validated_Batch_Files'
training_bad_file_path = '/workspaces/automated-semiconductor-fault-detection/Bad_Batch_Files'
training_column_count = 592
prediction_column_count = 591

pattern = r"^wafer_\d{8}_\d{6}.csv"
regex = re.compile(pattern)

if os.path.exists(training_good_file_path):
    shutil.rmtree(training_good_file_path)
    print('Previous run folder got deleted')
os.makedirs(training_good_file_path)
print('new good files folder created with deletion', training_good_file_path)

if os.path.exists(training_bad_file_path):
    shutil.rmtree(training_bad_file_path)
    print('Previous bad run folder got deleted')
os.makedirs(training_bad_file_path)
print('new bad files folder created', training_bad_file_path)
    
for i in training_batch_files_folder:
    if re.match(regex,i):
        with open(os.path.join(training_batch_file_directory,i),'r',encoding='utf-8') as file:
            reader = csv.reader(file)
            first_row = next(reader)
            if training_column_count == len(first_row):
                training_source_path = os.path.join(training_batch_file_directory,i)
                training_destination_path = os.path.join(training_good_file_path,i)
                shutil.copy2(training_source_path,training_destination_path)
                #print(f'Files copied {i}')
                #print(f'Files copied')
            else:
                training_source_path = os.path.join(training_batch_file_directory,i)
                training_destination_path = os.path.join(training_bad_file_path,i)
                shutil.copy2(training_source_path,training_destination_path)
                #print(f'Correct Name but missing columns Files copied {i}')
                #print(f'Correct Name but missing columns Files copied')
    else:
        training_source_path = os.path.join(training_batch_file_directory,i)
        training_destination_path = os.path.join(training_bad_file_path,i)
        shutil.copy2(training_source_path,training_destination_path)
        #print(f'Bad File copied {i}')
        #print(f'Bad File copied')