from decimal import Decimal, getcontext
from vector import Vector
import matplotlib.pyplot as plt
import numpy as np

getcontext().prec = 30


class Line(object):

    NO_NONZERO_ELTS_FOUND_MSG = 'No nonzero elements found'

    def __init__(self, normal_vector=None, constant_term=None):
        self.dimension = 2

        if not normal_vector:
            all_zeros = ['0']*self.dimension
            normal_vector = Vector(all_zeros)
        self.normal_vector = normal_vector

        if not constant_term:
            constant_term = Decimal('0')
        self.constant_term = Decimal(constant_term)

        self.set_basepoint()


    def set_basepoint(self):
        try:
            n = self.normal_vector
            c = self.constant_term
            basepoint_coords = ['0']*self.dimension

            initial_index = Line.first_nonzero_index(n.coordinates)
            initial_coefficient = n.coordinates[initial_index]

            basepoint_coords[initial_index] = c/initial_coefficient
            self.basepoint = Vector(basepoint_coords)

        except Exception as e:
            if str(e) == Line.NO_NONZERO_ELTS_FOUND_MSG:
                self.basepoint = None
            else:
                raise e


    def __str__(self):

        num_decimal_places = 3

        def write_coefficient(coefficient, is_initial_term=False):
            coefficient = round(coefficient, num_decimal_places)
            if coefficient % 1 == 0:
                coefficient = int(coefficient)

            output = ''

            if coefficient < 0:
                output += '-'
            if coefficient > 0 and not is_initial_term:
                output += '+'

            if not is_initial_term:
                output += ' '

            if abs(coefficient) != 1:
                output += '{}'.format(abs(coefficient))

            return output

        n = self.normal_vector

        try:
            initial_index = Line.first_nonzero_index(n.coordinates)
            terms = [write_coefficient(n[i], is_initial_term=(i==initial_index)) + 'x_{}'.format(i+1)
                     for i in range(self.dimension) if round(n[i], num_decimal_places) != 0]
            output = ' '.join(terms)

        except Exception as e:
            if str(e) == self.NO_NONZERO_ELTS_FOUND_MSG:
                output = '0'
            else:
                raise e

        constant = round(self.constant_term, num_decimal_places)
        if constant % 1 == 0:
            constant = int(constant)
        output += ' = {}'.format(constant)

        return output


    @staticmethod
    def first_nonzero_index(iterable):
        for k, item in enumerate(iterable):
            if not MyDecimal(item).is_near_zero():
                return k
        raise Exception(Line.NO_NONZERO_ELTS_FOUND_MSG)

    def graph_range(self, range): 
        A = self.normal_vector.coordinates[0]
        B = self.normal_vector.coordinates[1]
        k = Decimal(self.constant_term)
        normal_vector = np.array([[0,0,A,B]])
        X,Y,U,V = zip(*normal_vector)        
        
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xticks(np.arange(-50, 50, 1))
        ax.set_yticks(np.arange(-50, 50., 1))

        try:
            #y = (k - (A*x)) / B
            y = [(k-(A*x))/B for x in range]                                 
            
            plt.quiver(X,Y,U,V, angles='xy', scale_units='xy', scale=1)
            plt.plot(range, y)              
            print(self.basepoint)
            plt.scatter([self.basepoint.coordinates[0]], [[self.basepoint.coordinates[1]]], c = 'r')
            plt.grid()                                    
            plt.show()
        except Exception as e:
            print(e)
    
    def plot_against(self, l2, range):
        #Get the array outputs for y1 = F1(x1) y2 = F2(x2)
        #line1
        A1 = self.normal_vector.coordinates[0]
        B1 = self.normal_vector.coordinates[1]
        k1 = Decimal(self.constant_term)
        normal1 = np.array([[0,0,A1,B1]], dtype=float)
        #line2 
        A2 = l2.normal_vector.coordinates[0]
        B2 = l2.normal_vector.coordinates[1]
        k2 = Decimal(l2.constant_term)
        normal2 = np.array([[0,0,A2,B2]], dtype=float)

        #plot grid
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xticks(np.arange(-50, 50, 1))
        ax.set_yticks(np.arange(-50, 50., 1))
        plt.grid()
        
        try:

            #Line equation: y = (k - (A*x)) / B
            y1 = [(k1-(A1*x))/B1 for x in range]                                 
            y2 = [(k2-(A2*x))/B2 for x in range]                                                         
            plt.plot(range, y1, c='g')                                                                
            plt.plot(range, y2, c='b')      

            #plot normal vectors            
            X,Y,U,V = zip(*normal1)        
            plt.quiver(X,Y,U,V, angles='xy', scale_units='xy', scale=1, color='g')
            X,Y,U,V = zip(*normal2)        
            plt.quiver(X,Y,U,V, angles='xy', scale_units='xy', scale=1, color='b')

            #plot base points
            plt.scatter([self.basepoint.coordinates[0]], [[self.basepoint.coordinates[1]]], c = 'r')
            plt.scatter([l2.basepoint.coordinates[0]], [[self.basepoint.coordinates[1]]], c = 'r')

            plt.show()

        except Exception as e:
            print(e)

    def is_parallel_to(self, l2):                
        return self.normal_vector.is_parallel_to(l2.normal_vector)


class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps