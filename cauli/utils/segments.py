class Segments(object):
    def find_continous_segments(self,array):
        """Find the continous segments here"""
        prev = array[0]
        segments = []
        start = array[0]
        end = array[0]
        for now in array[1:]:
            if now == prev + 1:
                end = now
            else:
                each_segment = [start,end]
                segments.append(each_segment)
                start = now
                end = now
            prev = now
        each_segment = [start, end]
        segments.append(each_segment)
        return segments

# c = Segments()
# print(c.find_continous_segments([0, 1, 2, 3, 6, 7, 8, 12, 13, 20, 21]))