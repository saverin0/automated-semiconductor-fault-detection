import os
import re
import shutil
import csv


# training_batch_file_directory = '/workspaces/automated-semiconductor-fault-detection/Training_Batch_Files'
# training_batch_files_folder = os.listdir(training_batch_file_directory)
# training_good_file_path = '/workspaces/automated-semiconductor-fault-detection/Validated_Batch_Files'
# training_bad_file_path = '/workspaces/automated-semiconductor-fault-detection/Bad_Batch_Files'
# training_column_count = 592
# prediction_column_count = 591

# pattern = r"^wafer_\d{8}_\d{6}.csv"
# regex = re.compile(pattern)

# if os.path.exists(training_good_file_path):
#     shutil.rmtree(training_good_file_path)
#     print('Previous run folder got deleted')
# os.makedirs(training_good_file_path)
# print('new good files folder created with deletion', training_good_file_path)

# if os.path.exists(training_bad_file_path):
#     shutil.rmtree(training_bad_file_path)
#     print('Previous bad run folder got deleted')
# os.makedirs(training_bad_file_path)
# print('new bad files folder created', training_bad_file_path)

class FileValidator:
    def __init__(self,input_dir,good_dir,bad_dir,expected_columns,log_file, filename_pattern):
        self.input_dir = input_dir
        self.good_dir = good_dir #directory where good batch files will be saved 
        self.bad_dir = bad_dir #directory where bad batch files will be saved
        self.expected_columns = expected_columns #number of expected columns in prediction and training json files
        self.log_file = log_file #log file
        self.regex = re.compile(filename_pattern) #using re.compile for effciency 
        self.good_files = 0 # for logging  
        self.bad_files = 0  # for logging  


    def setup_directories(self):
        for path in [self.good_dir,self.bad_dir]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)
        
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def log(self,message):
        with open(self.log_file,'a') as log:
            log.write(message + '\n')

    def validate_file(self):
        self.setup_directories()
        files = os.listdir(self.input_dir)

        for i in files:
            full_path = os.path.join(self.input_dir,i)
            try:
                if self.regex.match(i):
                    with open(full_path,'r',encoding='utf-8') as f:
                        reader = csv.reader(f)
                        first_row = next(reader)

                    if len(first_row) == self.expected_columns:
                        self.copy_file(full_path,self.good_dir,i)
                        self.good_files += 1
                        self.log(f"[Good file] {i}: Column count OK")
                    else:
                        self.copy_file(full_path,self.bad_dir,i)
                        self.bad_files += 1
                        self.log(f"[Bad file] {i}: Column count mismatch. Expected {self.expected_columns}, found {len(first_row)}")
                else:
                     self.copy_file(full_path,self.bad_dir,i)
                     self.bad_files += 1
                     self.log(f"[Bad] {i}: Invalid filname format")
            except Exception as e:
                self.copy_file(full_path,self.bad_dir,i)
                self.bad_files += 1
                self.log(f"[Error] {i}: {e}")


    def copy_file(self,src,dst_dir,filename):
        shutil.copy2(src,os.path.join(dst_dir,filename))

    
    def summary(self):
        print(f"\n Validation completed:")
        print(f" {self.good_files} good file(s)")
        print(f"{self.bad_files} bad file(s)")
        print(f"logs saved at: {self.log_file} ")
    
# for i in training_batch_files_folder:
#     if re.match(regex,i):
#         with open(os.path.join(training_batch_file_directory,i),'r',encoding='utf-8') as file:
#             reader = csv.reader(file)
#             first_row = next(reader)
#             if training_column_count == len(first_row):
#                 training_source_path = os.path.join(training_batch_file_directory,i)
#                 training_destination_path = os.path.join(training_good_file_path,i)
#                 shutil.copy2(training_source_path,training_destination_path)
#                 #print(f'Files copied {i}')
#                 #print(f'Files copied')
#             else:
#                 training_source_path = os.path.join(training_batch_file_directory,i)
#                 training_destination_path = os.path.join(training_bad_file_path,i)
#                 shutil.copy2(training_source_path,training_destination_path)
#                 #print(f'Correct Name but missing columns Files copied {i}')
#                 #print(f'Correct Name but missing columns Files copied')
#     else:
#         training_source_path = os.path.join(training_batch_file_directory,i)
#         training_destination_path = os.path.join(training_bad_file_path,i)
#         shutil.copy2(training_source_path,training_destination_path)
#         #print(f'Bad File copied {i}')
#         #print(f'Bad File copied')


if __name__ == "__main__":
    Training_Files_Validator = FileValidator(
        input_dir= '/workspaces/automated-semiconductor-fault-detection/Training_Batch_Files',
        good_dir = '/workspaces/automated-semiconductor-fault-detection/Validated_Batch_Files',
        bad_dir= '/workspaces/automated-semiconductor-fault-detection/Bad_Batch_Files',
        expected_columns= 592,
        log_file="/workspaces/automated-semiconductor-fault-detection/training_validation_log.txt",
        filename_pattern=r"^wafer_\d{8}_\d{6}.csv"
)
Training_Files_Validator.validate_file()
Training_Files_Validator.summary()