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

    def normalize_zscore(self):
        for ix in range(self.row):
            std_val = np.std(self.mts_org[ix, :])

            if std_val == 0:
                if self.mts_org[ix, 0] != 0:
                    self.mts_org[ix, :] = 1
            else:
                self.mts_org[ix, :] = stats.zscore(self.mts_org[ix, :])
