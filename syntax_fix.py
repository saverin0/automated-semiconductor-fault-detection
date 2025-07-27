with open('src/prediction/predictor.py', 'r') as file:
    content = file.read()

# Fix the except statement
fixed_content = content.replace('except\n            \n         Exception', 'except Exception')

with open('src/prediction/predictor.py', 'w') as file:
    file.write(fixed_content)

print("Syntax error fixed!")
