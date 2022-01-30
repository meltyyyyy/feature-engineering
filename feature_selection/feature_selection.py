from feature_selection.iterative_selection.iterative_selection import IterativeSelection
from feature_selection.model_based_selection.model_based_selection import ModelBasedSelection
from feature_selection.univariate_statistics.univariate_statistics import UnivariateStatistics


def execute(arg):
    if arg == 'univariate_statistics':
        univariate_statistics = UnivariateStatistics()
        univariate_statistics.main()
    elif arg == 'model_based_selection':
        model_based_selection = ModelBasedSelection()
        model_based_selection.main()
    elif arg == 'iterative_selection':
        iterative_selection = IterativeSelection()
        iterative_selection.main()
    else:
        print("Cannot execute feature selection method.")
