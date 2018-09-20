import datetime
import numpy
import timeit
from cauli.test.find_intervals import *

#
# d1 = [[8.133, 8.732999999999999], [11.9, 12.9], [31.533, 33.0], [37.167, 38.433], [52.266999999999996, 53.333], [53.367, 53.567], [57.233000000000004, 58.367], [59.4, 60.933], [60.967, 62.433], [66.2, 68.3], [68.1, 70.933], [70.967, 73.233], [79.667, 83.76700000000001], [85.9, 86.06700000000001], [86.03299999999999, 87.7], [88.8, 91.06700000000001], [92.53299999999999, 95.46700000000001], [99.833, 101.2], [104.133, 104.7], [106.56700000000001, 107.633], [110.03299999999999, 110.93299999999999], [110.867, 112.667], [112.633, 114.23299999999999], [116.1, 122.9], [123.0, 123.1], [125.06700000000001, 126.26700000000001], [130.36700000000002, 131.167], [132.3, 137.7], [142.7, 146.267], [151.3, 153.667], [159.967, 164.6], [167.967, 172.233], [182.4, 185.467], [190.9, 191.933], [197.5, 199.667], [204.067, 204.933], [209.233, 210.333], [210.4, 211.733], [211.833, 212.8], [215.1, 223.7], [215.833, 216.13299999999998], [217.86700000000002, 218.13299999999998], [227.733, 228.1], [235.467, 239.3], [239.233, 239.7], [252.167, 257.933], [264.467, 269.867], [269.9, 271.233], [275.367, 277.9], [290.33299999999997, 296.5], [301.1, 303.867], [305.767, 310.267], [312.7, 315.233], [315.367, 317.66700000000003], [322.033, 324.033], [328.833, 329.733], [329.967, 338.833], [341.467, 343.86699999999996], [345.9, 354.667], [365.7, 367.1], [366.533, 367.56699999999995], [370.0, 373.767], [373.733, 374.1], [374.833, 383.86699999999996], [383.733, 385.8], [385.93300000000005, 386.13300000000004], [388.333, 389.63300000000004], [390.6, 391.233], [392.667, 395.2], [397.2, 400.3], [400.233, 402.267], [402.2, 406.833], [406.8, 409.233], [418.06699999999995, 419.667], [426.1, 430.36699999999996], [432.033, 434.667], [434.56699999999995, 437.43300000000005], [444.167, 445.533], [445.43300000000005, 452.7], [455.56699999999995, 456.233], [456.167, 461.833]]
# d2 = [[3.867, 5.167000000000001], [7.132999999999999, 8.1], [13.5, 14.267000000000001], [68.733, 69.3], [70.967, 73.233], [79.667, 84.167], [88.8, 90.0], [96.4, 98.56700000000001], [101.3, 104.7], [106.56700000000001, 107.167], [107.03299999999999, 107.8], [172.267, 172.7], [173.767, 174.7], [212.833, 215.067], [223.733, 227.63299999999998], [228.63299999999998, 229.967], [229.63299999999998, 235.433], [257.967, 259.5], [339.63300000000004, 340.9], [340.733, 341.333], [363.767, 365.667], [368.267, 369.7], [391.333, 392.63300000000004], [409.267, 410.1], [411.167, 411.533], [411.833, 413.36699999999996], [413.4, 413.533], [413.767, 414.8], [420.36699999999996, 421.9], [421.93300000000005, 426.06699999999995], [430.767, 432.0], [437.467, 441.63300000000004]]
#
# def get_two_lists():
#     d1 = numpy.random.randint(100, size=(1000, 2))
#     d2 = numpy.random.randint(100, size=(1000, 2))
#     return numpy.sort(d1),numpy.sort(d2)
#
#
# # code snippet to be executed only once
#
# d1,d2 = get_two_lists()
#
# c = numpy.array([[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]])
# # c = numpy.random.randint(2, size=(2, 40))
# print(c.dtype)
# c = c.all(axis=0)
#
# x = numpy.where(c)[False]
#
# import timeit
# print("------------------ Simple 'AND' -------------------")
# print("Time taken to and two binaries :",timeit.timeit('a = 0b0000000001111111111100000001111111111; b = 0b0000000001111111111100000001111111111; c = a&b',number=10000))
# print("------------------ Bitmap method -------------------")
# print("Time taken to load data :",timeit.timeit(lambda : numpy.random.randint(2, size=(2, 1000)),number=10000))
# print("Time taken to perform 'AND' : ",timeit.timeit(lambda :c.all(axis=0),number=10000))
# print("Time taken to find indexes of intersections : ",timeit.timeit(lambda : numpy.where(c)[False],number=10000))
#
# print("------------------ Linear method -------------------")
# print("Time taken to load data :",timeit.timeit(lambda : get_two_lists(),number=10000))
# print("Time taken to compute intersection : ", timeit.timeit(lambda: return_intersections(d1,d2),number=10000))





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
# mysetup = '''
# import numpy as np
# from find_intervals import return_intersections,get_two_lists
# '''
#
# # code snippet whose execution time is to be measured
# mycode = '''
# d1,d2 = get_two_lists()
# d = return_intersections(d1,d2)
# '''
#
# # timeit statement
# print("Time taken to compute intersection in Linear method :",
# timeit.timeit(setup=mysetup,
#               stmt=mycode,
#               number=10000))
#
mysetup = '''
import numpy as np
from find_intervals import return_intersections
'''

# code snippet whose execution time is to be measured
mycode = '''
a = 0b0000000001111111111100000001111111111
b = 0b0000000001111111111100000001111111111
i=0
data = list()
for c in a&b:
    if(c):
        data.append(i)
    i=i+1
'''

# timeit statement
print("Time taken by sort method alone :",
timeit.timeit(setup=mysetup,
              stmt=mycode,
              number=10000))
#
