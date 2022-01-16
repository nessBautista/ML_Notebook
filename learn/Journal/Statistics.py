class Combination:
    """Represents the Combination operation.
    Useful to find the number of combinations within a Event Space
    """

    def __init__(self, space_size, events_number):
        self.space_size = space_size
        self.events_number = events_number

    def combination(self):
        n_fac = self.__factorial(self.space_size)
        r_fac = self.__factorial(self.events_number)
        numerator = n_fac
        divisor = r_fac * (self.__factorial(self.space_size-self.events_number))
        return (numerator/divisor)

    def __factorial(self, n):
        if n < 0:
            print('There is no factorial for 0')
        if n == 0:
            return 1
        else:
            factorial = 1
            while(n>1):
                factorial *= n
                n -=1
        return factorial
    
    
        
    
comb1 = Combination(space_size=3, events_number=0)
comb2 = Combination(space_size=17, events_number=2)
comb3 = Combination(space_size=20, events_number=2)
print(f'{(comb1.combination()*comb2.combination())/comb3.combination()}={68/95}')
comb4 = Combination(space_size=3, events_number=3)
print(comb3.combination())