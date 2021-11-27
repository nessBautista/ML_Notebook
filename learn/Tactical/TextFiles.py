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

try_read_non_existing_file()

filepath="./learn/Tactical/sample.txt"
write_to_file("Hello", filepath=filepath)
read_file(filepath)
append_to_file("hello2", filepath)
append_to_file("hello3", filepath)
print("---after appending")
read_file(filepath)