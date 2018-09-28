import datetime
import numpy
import timeit
from cauli.test.find_intervals import *

#
# d1 = [[8.133, 8.732999999999999], [11.9, 12.9], [31.533, 33.0], [37.167, 38.433], [52.266999999999996, 53.333], [53.367, 53.567], [57.233000000000004, 58.367], [59.4, 60.933], [60.967, 62.433], [66.2, 68.3], [68.1, 70.933], [70.967, 73.233], [79.667, 83.76700000000001], [85.9, 86.06700000000001], [86.03299999999999, 87.7], [88.8, 91.06700000000001], [92.53299999999999, 95.46700000000001], [99.833, 101.2], [104.133, 104.7], [106.56700000000001, 107.633], [110.03299999999999, 110.93299999999999], [110.867, 112.667], [112.633, 114.23299999999999], [116.1, 122.9], [123.0, 123.1], [125.06700000000001, 126.26700000000001], [130.36700000000002, 131.167], [132.3, 137.7], [142.7, 146.267], [151.3, 153.667], [159.967, 164.6], [167.967, 172.233], [182.4, 185.467], [190.9, 191.933], [197.5, 199.667], [204.067, 204.933], [209.233, 210.333], [210.4, 211.733], [211.833, 212.8], [215.1, 223.7], [215.833, 216.13299999999998], [217.86700000000002, 218.13299999999998], [227.733, 228.1], [235.467, 239.3], [239.233, 239.7], [252.167, 257.933], [264.467, 269.867], [269.9, 271.233], [275.367, 277.9], [290.33299999999997, 296.5], [301.1, 303.867], [305.767, 310.267], [312.7, 315.233], [315.367, 317.66700000000003], [322.033, 324.033], [328.833, 329.733], [329.967, 338.833], [341.467, 343.86699999999996], [345.9, 354.667], [365.7, 367.1], [366.533, 367.56699999999995], [370.0, 373.767], [373.733, 374.1], [374.833, 383.86699999999996], [383.733, 385.8], [385.93300000000005, 386.13300000000004], [388.333, 389.63300000000004], [390.6, 391.233], [392.667, 395.2], [397.2, 400.3], [400.233, 402.267], [402.2, 406.833], [406.8, 409.233], [418.06699999999995, 419.667], [426.1, 430.36699999999996], [432.033, 434.667], [434.56699999999995, 437.43300000000005], [444.167, 445.533], [445.43300000000005, 452.7], [455.56699999999995, 456.233], [456.167, 461.833]]
# d2 = [[3.867, 5.167000000000001], [7.132999999999999, 8.1], [13.5, 14.267000000000001], [68.733, 69.3], [70.967, 73.233], [79.667, 84.167], [88.8, 90.0], [96.4, 98.56700000000001], [101.3, 104.7], [106.56700000000001, 107.167], [107.03299999999999, 107.8], [172.267, 172.7], [173.767, 174.7], [212.833, 215.067], [223.733, 227.63299999999998], [228.63299999999998, 229.967], [229.63299999999998, 235.433], [257.967, 259.5], [339.63300000000004, 340.9], [340.733, 341.333], [363.767, 365.667], [368.267, 369.7], [391.333, 392.63300000000004], [409.267, 410.1], [411.167, 411.533], [411.833, 413.36699999999996], [413.4, 413.533], [413.767, 414.8], [420.36699999999996, 421.9], [421.93300000000005, 426.06699999999995], [430.767, 432.0], [437.467, 441.63300000000004]]

def get_two_lists():
    d1 = numpy.random.randint(100, size=(10, 2))
    d2 = numpy.random.randint(100, size=(10, 2))
    return numpy.sort(d1),numpy.sort(d2)
#
#
# # code snippet to be executed only once
#
d1,d2 = get_two_lists()
#
c = numpy.array([[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]])
# # c = numpy.random.randint(2, size=(2, 40))
# print(c.dtype)
# c = c.all(axis=0)
#
# x = numpy.where(c)[False]
#
from bitstring import BitArray; c = BitArray([1,1,0,0,1]) ; d = BitArray([1,1,0,0,1]);
c=0b11100
print(type(c))
print(type(d))
d = 0b11000
print("Time taken to perform and operation :",timeit.timeit('x = 0b11111100000001111001; d = 0b11111100000001111001;c = x & d',number=10000))  #100
print("Time taken to perform xor operation :",timeit.timeit('x = 0b11111100000001111001; d = 0b11111100000001111001;c = x ^ d',number=10000))  #100
print("Time taken to perform left shift operation :",timeit.timeit('x = 0b11111100000001111001; c = x << 1',number=10000))  #100
print("Time taken to perform binary comparison :",timeit.timeit('x = 0b11111100000001111001; d = 0b11111100000001111001; c = 1 if (x>d) else 0',number=10000))  #100
# print("Time taken to join list into string binaries :",timeit.timeit('import numpy as np;x = np.random.randn(10).tolist(); x = ",".join(map(str, x))',number=10000))
print("Time taken to convert string into int :",timeit.timeit("x = int('111000',2)", number=10000))
print("Time taken to join list into string binaries and compute and :",timeit.timeit(lambda: c & d, number=10000))

# print("Time taken to multiply numbers :",timeit.timeit('134213412431243123413412*101312341324*12412431241243*12341431341341341',number=10000))
# print("Time taken to and two binaries2 :",timeit.timeit('from functools import reduce;from operator import mul  ;c = reduce(mul, [4,5])',number=10000))
# print("Time taken to find indexes :",timeit.timeit("[i for i, ltr in enumerate('oottaaoo') if ltr == 'o']",number=10000))

# import timeit
# print("------------------ Simple 'AND' -------------------")
# print("Time taken to and two binaries :",timeit.timeit('a = 0b0000000001111111111100000001111111111; b = 0b0000000001111111111100000001111111111; c = a&b',number=10000))
# print("------------------ Bitmap method -------------------")
# print("Time taken to load data :",timeit.timeit(lambda : numpy.random.randint(2, size=(2, 1000)),number=10000))
# print("Time taken to perform 'AND' : ",timeit.timeit(lambda :c.all(axis=0),number=10000))
# print("Time taken to find indexes of intersections using numpy : ",timeit.timeit(lambda : numpy.where(c)[False],number=10000))
# #




#
# # code snippet to be executed only once
# mysetup = "import numpy"
#
# # code snippet whose execution time is to be measured
# mycode = '''
# c = numpy.random.randint(2, size=(2, 1000))
#
# # c = c.all(axis=0)
# # x = numpy.where(c)[False]
# '''
#
# # timeit statement
# print("Time taken to compute intersection in Bitmap method : ",
# timeit.timeit(setup=mysetup,
#               stmt=mycode,
#               number=10000))
#
#
#
# print("using lambda and mul :",
#       timeit.timeit('from functools import reduce;from operator import mul;c = reduce(mul, [4,5,6,7,8,9,10])'))
#
# print("using lambda :",
#       timeit.timeit('from functools import reduce;from operator import mul;c = reduce(lambda x, y: x * y, [4,5,6,7,8,9,10])'))

mysetup = '''
import numpy as np
from find_intervals import return_intersections 
'''

# code snippet whose execution time is to be measured
mycode = '''
x = [42341234113412341234124312341234123412,512341234123132412431234123412341234123421]
prod = 1
i = 0
while True:
    prod = prod * x[i]
    i = i + 1
    if i == len(x):
        break
'''

# timeit statement
# print("Time taken to multiply numbers in a list :",
# timeit.timeit(setup=mysetup,
#               stmt=mycode,number=10000))


def find_offsets(haystack, needle):
    """
    Find the start of all (possibly-overlapping) instances of needle in haystack
    """
    offs = -1
    while True:
        offs = haystack.find(needle, offs+1)
        if offs == -1:
            break
        else:
            yield offs

def find_offsets1(haystack):
    """
    Find the start of all (possibly-overlapping) instances of needle in haystack
    """
    offs = -1
    while True:
        offs = offs + 1
        if not haystack:
            break
        if haystack & 1:
            yield offs
        haystack = haystack >> 1


r = BitArray('0b11110000111')
print("Find offset ",list(find_offsets1(int("11110000111",2))))
print("Time taken to find indices using bit array:",timeit.timeit(lambda: r.findall([1]), number=10000))
print("Time taken to find indices : ",timeit.timeit(lambda: find_offsets("11110000111","1"), number=10000))
print("Time taken to find indices fast : ",timeit.timeit(lambda: find_offsets1(int("11110000111",2)), number=10000))
print("Time taken to convert int to binary string:",timeit.timeit('bin(62342421431234)[2:] ',number=10000))

# mysetup = '''
# import numpy as np
#
# '''
#
# # code snippet whose execution time is to be measured
# mycode = '''
#
# data = list()
# Seg
# '''
#
# # timeit statement
# print("Time taken by sort method alone :",
# timeit.timeit(setup=mysetup,
#               stmt=mycode,
#               number=10000))
# #
array =[0,2,3]
l = [[1,2],[4,5],[5,6],[7,9]]
from cauli.utils.segments import Segments
print("Time taken to compute actual intervals :",timeit.timeit(lambda : Segments().find_continous_segments(array,l),number=10000))


print("------------------ Linear method -------------------")
print("Time taken to load data :",timeit.timeit(lambda : get_two_lists(),number=10000))
print("Time taken to compute intersection : ", timeit.timeit(lambda: return_intersections(d1,d2),number=10000))


person1 = [[2,4],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,18],[19,20],[21,22],[23,24],[25,26],[27,28],[29,30],[31,32],[33,34],[35,36],[37,38],[39,40],[41,53],[56,57]]
person2 = [[2,4],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,18],[19,20],[21,22],[23,24],[25,26],[27,28],[29,30],[31,32],[33,34],[35,36],[37,38],[39,40],[41,53],[56,57]]


# person1 = [[2,],[6,7]]
# person2 = [[5,5],[6,7]]

def xy():
    if person2[-1][0] > person1[0][1]:
        return 1
    else:
        return 0

d1 = numpy.random.randint(100, size=(10, 2))
d2 = numpy.random.randint(100, size=(10, 2))
d3 = numpy.random.randint(100, size=(10, 2))
d4 = numpy.random.randint(100, size=(10, 2))
d5 = numpy.random.randint(100, size=(10, 2))

def n_2():
    return return_intersections(d1,d2)

def n_3():
    d4 = return_intersections(d1,d2)
    return return_intersections(d4,d3)

def n_4():
    n3 = return_intersections(d1,d2)
    n4 = return_intersections(n3,d3)
    return return_intersections(n4,d4)

def n_5():
    n3 = return_intersections(d1,d2)
    n4 = return_intersections(n3,d3)
    n5 = return_intersections(n4,d4)
    return return_intersections(n5,d5)
def checkNext(p1,p2):
    p1 = list(map(lambda x: x[1], p1))
    p2 = list(map(lambda x : x[0], p2))
    for each_t in p1:
        if each_t in p2:
            return True
    return False
# print(checkNext(person1,person2))
# def checkNext(p1,p2):
#     p1 = p1[:,1]
#     p2 = p2[:,0]
#     for each_t in p1:
#         if each_t in p2:
#             return True
#     return False

print("Time taken to compute next query :",timeit.timeit(lambda : checkNext(person1,person2),number=10000))