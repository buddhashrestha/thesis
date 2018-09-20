'''
reference : https://codereview.stackexchange.com/questions/178427/given-2-disjoint-sets-of-intervals-find-the-intersections

'''

import pandas as pd
import numpy
import ast
def getIntersection(interval_1, interval_2):
    start = max(interval_1[0], interval_2[0])
    end = min(interval_1[1], interval_2[1])
    if start < end:
        return (start, end)
    return None

def return_intersections(intervals1, intervals2=[]):
    iter1 = iter(intervals1)
    iter2 = iter(intervals2)

    interval1 = next(iter1)
    interval2 = next(iter2)

    while True:
        intersection = getIntersection(interval1, interval2)
        if intersection:
            yield intersection
            try:
                if intersection[1] == interval1[1]:
                    interval1 = next(iter1)
                else:
                    interval2 = next(iter2)
            except StopIteration:
                return

        try:
            while interval1[0] > interval2[1]:
                interval2 = next(iter2)
            while interval2[0] > interval1[1]:
                interval1 = next(iter1)
        except StopIteration:
            return


#
# set1 = [(1, 6), (10, 15), (20, 30)]
# set2=[(2, 5), (7, 13),(17,22),(29,35)]
# table = pd.read_csv("/home/buddha/thesis/cauli/data/"+str(2) + '/person_segment_naive.csv',sep='\t')
# x = table.loc[table['person'] == int(40), table.columns != 'person']['segment'].iloc[0]
# y = table.loc[table['person'] == int(18), table.columns != 'person']['segment'].iloc[0]
# x = ast.literal_eval(x)
# x = numpy.array(x, dtype=float)
# y = ast.literal_eval(y)
# y = numpy.array(y, dtype=float)
# print("Y: ",y)
# print("X: ",x)
# print(list(return_intersections(x, y)))