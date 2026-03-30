from . import plot
from .plot import plot_data_2d, plot_data_3d

from . import data
from .data import generate_separable_dataset, generate_non_separable_dataset

__all__ = [
	"plot_data_2d",
	"plot_data_3d",
	"generate_separable_dataset",
	"generate_non_separable_dataset"
]

