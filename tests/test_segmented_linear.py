import numpy as np
from segmented_linear.segmented_linear import fit_segmented_linear

def test_fit_segmented_linear():
    # Sample data
    x = np.sort(np.random.choice(20, 50, replace=True))
    y = [2 + 1.5 * xi + np.random.normal(0, 2) if xi <= 10 else 15 + 0.5 * xi + np.random.normal(0, 2) for xi in x]
    
    # Test with interval
    models, x_subsets, y_pred_subsets, combined_bic = fit_segmented_linear(x, y, interval_list=[10])
    assert len(models) == 2
    assert len(x_subsets) == 2
    assert len(y_pred_subsets) == 2
    
    # Test without interval
    models, x_subsets, y_pred_subsets, combined_bic = fit_segmented_linear(x, y)
    assert len(models) == 1
    assert len(x_subsets) == 1
    assert len(y_pred_subsets) == 1

