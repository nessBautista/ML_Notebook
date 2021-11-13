# ints is a set that is made of the numbers from 0 to 9
ints = {i for i in range(10)}
print(ints)
#out: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# Comprehension using Range with a condition Filter
evens = {i for i in range(10) if i%2 == 0}
print(evens)
#out: {0, 2, 4, 6, 8}

# Apply math function to values in range
squares = {i*i for i in range(10)}
print(squares)
# notice how this are not in order, because this is a set
# out: {0, 1, 64, 4, 36, 9, 16, 49, 81, 25}

# Sets eliminates duplicates from sets
sqrs = {i*i for i in range(-5,5)}
print(sqrs)
#out: {0, 1, 4, 9, 16, 25}

# set comprehension on a list
primes = [2,2,2,3,3,5,5,5,7,11,11,13,13,13,13]
primes_squared = {p*p for p in primes}
print(primes_squared)
#out: {4, 9, 169, 49, 121, 25}

# more complex expresions: quadratic  transformations
transformed = {(2*x*x + 5*x + 10) for x in primes}
print(transformed)
#out: {43, 143, 307, 85, 28, 413}

# Flatten list and eliminiate duplicates
# take nums, a 2-D array
nums = [[1,3],[2,3],[3,98],[76,1]]
flat_set = {col for rows in nums for col in rows}
print(flat_set)
#out: {1, 2, 3, 98, 76}  <-- this is a flat set

