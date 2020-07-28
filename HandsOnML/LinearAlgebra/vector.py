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


    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)


    def __eq__(self, v):
        return self.coordinates == v.coordinates

    
    def plus(self, v):
        # https://realpython.com/python-zip-function/#using-zip-in-python
        # takes in iterables as arguments and returns an iterator. 
        # This iterator generates a series of tuples containing elements from each iterable.
        new_coordinates = [x+y for y,x in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def minus(self, v):        
        new_coordinates = [x-y for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)
    
    def times_scalar(self, c):        
        new_coordinates = [Decimal(c)*x for x in self.coordinates]
        return Vector(new_coordinates)

    def magnitude(self):
        sqrSum = sum([x**2 for x in self.coordinates])
        return Decimal(sqrt(sqrSum))
    
    def normalized(self):
        try:
            m = self.magnitude()
            return self.times_scalar(Decimal(1.0)/m)            
        
        except ZeroDivisionError:
            raise Exception(self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG)


    def dot(self, v):
        return sum ([x*y for x,y in zip(self.coordinates, v.coordinates)])

    def angle(self, v, in_degrees=False):
        try:
            u1 = self.normalized()
            u2 = v.normalized()
            normDotProduct = round(u1.dot(u2), 7)
            angle_in_radians = acos(normDotProduct)
        
            if in_degrees:
                degrees_per_radian = 180. / pi
                return angle_in_radians * degrees_per_radian
            else:            
                return angle_in_radians
        
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception('Can not compute an angle with the zero vector')

    
    def is_ortoghonal_to(self, v, tolerance=1e-10):
        return abs(self.dot(v)) < tolerance

    def is_zero(self, tolerance=1e-10):
        return self.magnitude() < tolerance

    def is_parallel_to(self, v):                
        return self.is_zero() or v.is_zero() or round(self.angle(v),2) == 0 or round(self.angle(v),2) == round(pi,2)

