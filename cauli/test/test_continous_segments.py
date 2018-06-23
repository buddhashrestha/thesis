import unittest
import numpy
import sys
sys.path.append('../')
from ..utils.segments import Segments


class TestFileDescriptor(unittest.TestCase):
    def test_find_continous_segments(self):
        c = Segments()
        segments = c.find_continous_segments([2, 3, 4, 7, 8, 9])
        actual_segments = numpy.array([[2,4],[7,9]])
        self.assertEqual(segments, actual_segments)