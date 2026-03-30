# src/tutorial/plot/_utils.py

import numpy as np
import pandas as pd
import plotly

def extend_range(values, factor=0.05, min_span=1e-9):
	vmin, vmax = np.min(values), np.max(values)
	delta = vmax - vmin
	if delta == 0:
		delta = min_span
	return vmin - factor * delta, vmax + factor * delta


def compute_plot_bounds(X, features, factor=0.05):
	return {
		feature: extend_range(X[feature], factor=factor)
		for feature in features
	}


def get_color_map(labels, colors):
	return {label: colors[i % len(colors)] for i, label in enumerate(labels)}

