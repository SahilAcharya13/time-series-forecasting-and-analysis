import pandas as pd

def process_marks(file_path, output_path):
    data = pd.read_excel(file_path)
    data['total_marks'] = data['marks'].apply(lambda x: sum(map(int, x.split(','))))
    data.to_excel(output_path, index=False)

input_file = 'C:/Users/dell/Downloads/marks.xlsx'
output_file = 'C:/Users/dell/Downloads/marks.xlsx'
process_marks(input_file, output_file)

print(f"The processed file with total marks is saved at: {output_file}")
