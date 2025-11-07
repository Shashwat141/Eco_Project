import csv

input_file = "FULL_National Educational Attainment Classification (NEAC) and labour market status (full dataset).csv"   # path to your 4.3GB CSV
output_file = "first_11_rows.csv"

with open(input_file, "r", newline='', encoding='utf-8') as infile, \
     open(output_file, "w", newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for i, row in enumerate(reader):
        if i >= 11:
            break
        writer.writerow(row)

print("✅ First 11 rows written to", output_file)
