from csv import writer
import csv

with open('datasets-48149-87794-PJM_Load_hourly.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    next(csv_reader)
    with open('onlyNumbers.csv','a+',  newline='') as write_obj:
        for row in csv_reader:
            csv_writer = writer(write_obj)
            csv_writer.writerow([row[1]])
            line_count+=1
        print(f'Processed {line_count} lines.')