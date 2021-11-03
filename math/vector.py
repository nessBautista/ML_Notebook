from math import sqrt, acos, pi
from decimal import Decimal, getcontext

class Vector(object):
    
    CANNOT_NORMALIZE_ZERO_VECTOR_MSG = "can not normalize the zero vector"
    
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(Decimal(x) for x in coordinates)
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')
    


    def __eq__(self, v):
        return self.coordinates == v.coordinates
    
    def plus(self, v):
        # https://realpython.com/python-zip-function/#using-zip-in-python
        # takes in iterables as arguments and returns an iterator. 
        # This iterator generates a series of tuples containing elements from each iterable.
        new_coordinates = [x+y for y,x in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)


