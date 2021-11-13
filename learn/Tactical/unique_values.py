test_dict = {'gfg' : [5, 6, 7, 8],
             'is' : [10, 11, 7, 5],
             'best' : [6, 12, 10, 8],
             'for' : [1, 2, 5]}
  
# printing original dictionary
print("The original dictionary is : " + str(test_dict))
  

# each val in this iteration is a list of numbers
print("-------values--------")
for val in test_dict.values():
    print(val)
print("---------------------")
# Extract Unique values dictionary values
# Using set comprehension
res = {item for val in test_dict.values() for item in val}

print(f'as a set:{res}')
print(f'as a list:{list(res)}')
