from .types import NPArray
from multipledispatch import dispatch
from typing import Tuple, Optional
import matplotlib.pyplot as plt


@dispatch(NPArray, NPArray, NPArray, NPArray)
def plot_predictions(test_points: NPArray,
                     mean: NPArray,
                     lower_ci: NPArray,
                     upper_ci: NPArray,
                     ax = None,
                     train_points: Optional[Tuple[NPArray, NPArray]] = None,
                     legend: bool = False,
                     title: str = None):
    if ax is None:
        fig, ax = plt.subplots()
    if train_points:
        ax.plot(train_points[0], train_points[1], 'o', label='Training data', color='tab:blue')
    ax.plot(test_points, mean, label='Predictive mean', color='tab:orange')
    ax.fill_between(test_points.squeeze(),
            lower_ci.squeeze(),
            upper_ci.squeeze(),
            color='tab:blue',
            alpha=0.4,
            label='Predictive variance')
    ax.plot(test_points, upper_ci, color="tab:blue", linestyle="--")
    ax.plot(test_points, upper_ci, color="tab:blue", linestyle="--")
    if legend:
        ax.legend(loc='best')
    if title:
        ax.set_title(title)
    return ax


@dispatch(NPArray, NPArray, NPArray)
def plot_predictions(test_points: NPArray,
                     mean: NPArray,
                     variance: NPArray,
                     ax = None,
                     train_points: Optional[Tuple[NPArray, NPArray]] = None,
                     legend: bool = False,
                     title: str = None):
    return plot_predictions(test_points,
                            mean,
                            mean.ravel() - 1.96*variance.ravel(),
                            mean.ravel() + 1.96*variance.ravel(),
                            ax,
                            train_points,
                            legend,
                            title,
                            )
    # if ax is None:
    #     fig, ax = plt.subplots()
    # if train_points:
    #     ax.plot(train_points[0], train_points[1], 'o', label='Training data', color='tab:blue')
    # ax.plot(test_points, mean, label='Predictive mean', color='tab:orange')
    # ax.plot(test_points.squeeze(),
    #         mean.squeeze() - 1.96 * variance.squeeze(),
    #         mean.squeeze() + 1.96 * variance.squeeze(),
    #         color='tab:blue',
    #         alpha=0.4,
    #         label='Predictive variance')
    # ax.plot(test_points, mean + 1.96 * variance, color="tab:blue", linestyle="--")
    # ax.plot(test_points, mean - 1.96 * variance, color="tab:blue", linestyle="--")
    # if legend:
    #     ax.legend(loc='best')
    # if title:
    #     ax.set_title(title)
    # return ax
