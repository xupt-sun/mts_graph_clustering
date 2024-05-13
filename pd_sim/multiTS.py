'''
Class MTS
'''
import numpy as np
from scipy import stats

class MultiTS:
    def __init__(self, mid, label, vecs):
        # vecs: each row is a time series
        self.mid = mid
        self.label = label
        self.mts_org = np.array(vecs)
        [self.row, self.column] = vecs.shape
