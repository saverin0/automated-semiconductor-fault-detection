import json


create_json

class create_json_schema(create_json):
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
    with open('output.json','w') as file:
    json.dump(datalist_training,file,indent = 4)

    print('Data saved successfully')