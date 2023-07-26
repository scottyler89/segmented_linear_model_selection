import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def fit_segmented_linear(x, y, interval_list=[]):
    """
    Fit segmented linear models to data based on given intervals.

    Parameters
    ----------
    x : numpy.ndarray
        1D array of independent variable data.
    y : numpy.ndarray
        1D array of dependent variable data.
    interval_list : list, optional
        List of cut-off points to define the intervals for segmentation. 
        For example, [10, 20] will create three intervals: x <= 10, 10 < x <= 20, and x > 20.
        By default, fits a single linear model to all data.

    Returns
    -------
    models : list
        List of statsmodels OLS regression results for each interval.
    x_subsets : list
        List of x subsets corresponding to each interval.
    y_pred_subsets : list
        List of predicted y values for each interval based on the respective models.
    combined_bic : float
        Combined Bayesian Information Criterion (BIC) for all models.

    Examples
    --------
    >>> x = np.array([1,2,3,11,12,13])
    >>> y = np.array([2,4,6,12,14,16])
    >>> models, x_subsets, y_pred_subsets, combined_bic = fit_segmented_linear(x, y, interval_list=[10])
    """

    # Lists to store results
    models = []
    x_subsets = []
    y_pred_subsets = []

    # Determine the intervals for modeling
    if not interval_list:
        intervals = [(None, None)]
    else:
        intervals = [(None, interval_list[0])]
        for i in range(len(interval_list) - 1):
            intervals.append((interval_list[i], interval_list[i+1]))
        intervals.append((interval_list[-1], None))

    # Total log likelihood for combined BIC
    total_log_likelihood = 0

    # Loop over intervals and fit models
    for lower, upper in intervals:
        if lower is None and upper is not None:
            subset_mask = x <= upper
        elif upper is None and lower is not None:
            subset_mask = x > lower
        elif lower is not None and upper is not None:
            subset_mask = (x > lower) & (x <= upper)
        else:
            subset_mask = np.ones_like(x, dtype=bool)

        x_subset = x[subset_mask]
        y_subset = y[subset_mask]

        model = sm.OLS(y_subset, sm.add_constant(x_subset)).fit()

        models.append(model)
        x_subsets.append(x_subset)
        y_pred_subsets.append(model.predict(sm.add_constant(x_subset)))

        total_log_likelihood += model.llf

    # Calculate combined BIC
    n_params = sum([len(model.params) for model in models]) - len(models) + 1  # Adjusting for intercepts
    combined_bic = -2 * total_log_likelihood + n_params * np.log(len(x))

    return models, x_subsets, y_pred_subsets, combined_bic

    