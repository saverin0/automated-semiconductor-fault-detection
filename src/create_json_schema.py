import json

datalist = {
	"SampleFileName" : "wafer_01012025_120000.csv",
	"LengthOfDateStampInFile" : 8,
	"LengthOfTimeStampInFile" : 6,
	"NumberOfColumns" : 592,
	"columns" : {}
}
for i in range(1,592):
    datalist['columns']['wafer'] = "varchar"
    datalist['columns'][f'Sensor - {i}'] = " float"

datalist["columns"]["Output"] = "Integer"
#print(datalist)
with open('output.json','w') as file:
   json.dump(datalist,file,indent = 4)

print('Data saved successfully')
