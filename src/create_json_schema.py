import json

class create_json_schema:

    def __init__ (self,output_training = "schema_training.json",output_prediction="schema_prediction.json"):
        self.output = output_training
        self.output = output_prediction

    def generate_schema_trainig(self):
        datalist_training = {
            "SampleFileName" : "wafer_01012025_120000.csv",
            "LengthOfDateStampInFile" : 8,
            "LengthOfTimeStampInFile" : 6,
            "NumberOfColumns" : 592,
            "columns" : {}
        }
        for i in range(1,592):
            datalist_training['columns']['wafer'] = "varchar"
            datalist_training['columns'][f'Sensor - {i}'] = " float"

        datalist_training["columns"]["Output"] = "Integer"
        #print(datalist)
        with open('schema_training.json','w') as file:
            json.dump(datalist_training,file,indent = 4)
        print('Training json saved successfully')

    def generate_schema_prediction(self):
        datalist_prediction = {
            "SampleFileName" : "wafer_01012025_120000.csv",
            "LengthOfDateStampInFile" : 8,
            "LengthOfTimeStampInFile" : 6,
            "NumberOfColumns" : 591,
            "columns" : {}
        }
        for i in range(1,592):
            datalist_prediction['columns']['wafer'] = "varchar"
            datalist_prediction['columns'][f'Sensor - {i}'] = " float"

        #datalist_prediction["columns"]["Output"] = "Integer"
        #print(datalist)
        with open('schema_prediction.json','w') as file:
            json.dump(datalist_prediction,file,indent = 4)
        print('Prediction json saved successfully')



if __name__ == "__main__":
    schema_creator = create_json_schema(output_training="schema_training.json",output_prediction="schema_prediction.json")
    schema_creator.generate_schema_trainig()
    schema_creator. generate_schema_prediction()