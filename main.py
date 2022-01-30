import sys

from binning.binning import Binning
from feature_selection import feature_selection
from interaction_feature.interaction_feature import InteractionFeature
from one_hot_encoding.one_hot_encoding import OneHotEncoding
from polynomial.polynomiall import PolynomialFeature

if __name__ == '__main__':
    if sys.argv[1] == 'one_hot_encoding':
        one_hot_encoding = OneHotEncoding()
        one_hot_encoding.main()
    elif sys.argv[1] == 'binning':
        binning = Binning()
        binning.main()
    elif sys.argv[1] == 'interaction_feature':
        interaction_feature = InteractionFeature()
        interaction_feature.main()
    elif sys.argv[1] == 'polynomial_feature':
        interaction_feature = PolynomialFeature()
        interaction_feature.main()
    elif sys.argv[1] == 'feature_selection':
        feature_selection.execute(sys.argv[2])
    else:
        print('Invalid augment')
