#step 2
from csv import writer
import csv
import pandas as pd

def convert():
    df = pd.read_csv('onlyNumbers.csv')
    line_count=0
    with open('onlyNumbers.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        #FIRST: Name another csv file to store it
        with open('windowWith10Inputs.csv','a+',  newline='') as write_obj:
            for row in csv_reader:
                #SECOND: change parameter here if larger than three for window
                if((line_count-10)>0 ):
                    csv_writer = writer(write_obj)
                    #THIRD: change parameter here if larger than three for window
                    csv_writer.writerow([df.iloc[line_count-10][0],df.iloc[line_count-9][0],df.iloc[line_count-8][0],df.iloc[line_count-7][0],df.iloc[line_count-6][0],df.iloc[line_count-5][0],df.iloc[line_count-4][0],df.iloc[line_count-3][0],df.iloc[line_count-2][0],df.iloc[line_count-1][0],df.iloc[line_count][0]] )
                line_count+=1

convert()