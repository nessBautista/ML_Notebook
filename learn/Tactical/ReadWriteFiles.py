import csv

## Basics

def try_read_non_existing_file():
    try:
        with open("NonExistingFile.txt") as file:
            file.read()
    except:
        print("File Not Found")


def write_to_file(text, filepath):
    with open(filepath, mode='w') as file:
        file.write(text)

def read_file(filepath):
    with open(filepath) as file:
        print(file.read())

def append_to_file(text,filepath):
    try:
        with open(filepath, mode='a') as file:
            print("\n", file=file)
            print(text, file=file)
    except:
        print("Not able to append")

## Reading a CSV file
def read_csv_file(filepath):
    try:
        with open(filepath) as csvfile:
            csv_object = csv.reader(csvfile)
            separator = ','
            for row in csv_object:
                line = separator.join(row)
                print(line)
    except:
        print("Not able to read csv file")

try_read_non_existing_file()
filepath="./learn/Tactical/sample.txt"
write_to_file("Hello", filepath=filepath)
read_file(filepath)
append_to_file("hello2", filepath)
append_to_file("hello3", filepath)
print("---after appending")
read_file(filepath)
print("---csv file reader")

filepath = "./datasets/kaggle/pima-indians-diabetes.data.csv"
read_csv_file(filepath)
