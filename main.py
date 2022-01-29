import sys

from binning.binning import Binning
from one_hot_encoding.one_hot_encoding import OneHotEncoding

if __name__ == '__main__':
    if sys.argv[1] == 'one_hot_encoding':
        one_hot_encoding = OneHotEncoding()
        one_hot_encoding.main()
    elif sys.argv[1] == 'binning':
        binning = Binning()
        binning.main()
    else:
        print('Invalid augment')
