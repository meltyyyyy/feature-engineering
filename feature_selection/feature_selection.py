from feature_selection.univariate_statistics.univariate_statistics import UnivariateStatistics


def execute(arg):
    if arg == 'univariate_statistics':
        univariate_statistics = UnivariateStatistics()
        univariate_statistics.main()
    else:
        print("Cannot execute feature selection method.")
