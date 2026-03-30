import numpy as np
import pandas as pd
import plotly.graph_objects
from ._utils import extend_range, compute_plot_bounds, get_color_map


# ======================
# 2D HELPERS
# ======================

def create_meshgrid_2d(features, bounds, resolution=100):
	x_min, x_max = bounds[features[0]]
	y_min, y_max = bounds[features[1]]

	x_vals = np.linspace(x_min, x_max, resolution)
	y_vals = np.linspace(y_min, y_max, resolution)

	dx = x_vals[1] - x_vals[0]
	dy = y_vals[1] - y_vals[0]

	x_vals = np.linspace(x_min - dx, x_max + dx, resolution)
	y_vals = np.linspace(y_min - dy, y_max + dy, resolution)

	xx, yy = np.meshgrid(x_vals, y_vals)

	grid = pd.DataFrame(
		{features[0]: xx.ravel(), features[1]: yy.ravel()}
	)

	return xx, yy, x_vals, y_vals, grid


def create_class_traces_2d(X, y, features, labels, label_to_color):
	traces = []
	for label in labels:
		mask = y == label
		traces.append(plotly.graph_objects.Scatter(
			x=X[mask][features[0]],
			y=X[mask][features[1]],
			mode="markers",
			marker=dict(size=7, color=label_to_color[label]),
			name=f"Class {label}"
		))
	return traces


def create_decision_regions_2d(
	xx, yy, x_vals, y_vals, Z_class, labels, label_to_color
):
	label_to_int = {label: i for i, label in enumerate(labels)}
	Z_int = np.vectorize(label_to_int.get)(Z_class)

	colorscale = []
	for label, i in label_to_int.items():
		color = label_to_color[label]
		colorscale.append([i / len(labels), color])
		colorscale.append([(i + 1) / len(labels), color])

	return plotly.graph_objects.Contour(
		x=x_vals,
		y=y_vals,
		z=Z_int,
		colorscale=colorscale,
		showscale=False,
		opacity=0.2,
		contours=dict(
			start=0,
			end=len(labels) - 1,
			size=1,
			coloring="fill"
		),
		line=dict(width=0),
		name="Decision Regions"
	)


def create_decision_boundary_and_margins_2d(
	xx, yy, x_vals, y_vals, classifier, highlight_support
):
	traces = []

	if hasattr(classifier, "decision_function"):
		Z = classifier.decision_function(
			pd.DataFrame(
				np.c_[xx.ravel(), yy.ravel()], columns=[xx.shape[1],
				yy.shape[1]]
			)
		).reshape(xx.shape)

		# Decision boundary
		traces.append(plotly.graph_objects.Contour(
			x=x_vals,
			y=y_vals,
			z=Z,
			colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
			showscale=False,
			contours=dict(start=0, end=0, size=1),
			line=dict(color='black', width=2),
			name='Decision Boundary'
		))

		# Margins
		traces.append(plotly.graph_objects.Contour(
			x=x_vals,
			y=y_vals,
			z=Z,
			colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
			showscale=False,
			contours=dict(start=-1, end=1, size=2),
			line=dict(color='gray', width=1, dash='dash'),
			name='Margins'
		))

	if highlight_support and hasattr(classifier, "support_vectors_"):
		sv = classifier.support_vectors_
		traces.append(plotly.graph_objects.Scatter(
			x=sv[:, 0],
			y=sv[:, 1],
			mode="markers",
			marker=dict(
				size=12, color="black", symbol='circle-open', line=dict(width=2)
			),
			name='Support Vectors'
		))

	return traces


def create_layout_2d(features, bounds):
	x_min, x_max = bounds[features[0]]
	y_min, y_max = bounds[features[1]]

	return plotly.graph_objects.Layout(
		xaxis=dict(
			title=features[0],
			range=[x_min, x_max],
			showline=True,
			linecolor="black",
			ticks="inside",
			zeroline=False,
			showgrid=True,
			gridcolor="lightgray",
			gridwidth=1,
			tickmode="linear",
			tick0=0,
			dtick=1,
			anchor="y",
			position=0,
			layer="below traces"
		),
		yaxis=dict(
			title=features[1],
			range=[y_min, y_max],
			showline=True,
			linecolor="black",
			ticks="inside",
			zeroline=False,
			showgrid=True,
			gridcolor="lightgray",
			gridwidth=1,
			tickmode="linear",
			tick0=0,
			dtick=1,
			anchor="x",
			position=0,
			layer="below traces",
			scaleanchor="x",
			scaleratio=1
		),
		hovermode=False,
		dragmode="pan",
		plot_bgcolor="white",
		paper_bgcolor="white",
		margin=dict(l=40, r=40, t=40, b=40)
	)


# ======================
# MAIN ENTRY POINT
# ======================

def plot_data_2d(df, features, target, classifier=None, highlight_support=True):
	if len(features) != 2:
		raise ValueError("plot_data_2d requires exactly 2 features")

	X = df[features]
	y = df[target].values
	labels = np.unique(y)

	bounds = compute_plot_bounds(X, features)
	label_to_color = get_color_map(labels, plotly.colors.qualitative.Plotly)

	traces = create_class_traces_2d(X, y, features, labels, label_to_color)

	if classifier is not None:
		xx, yy, x_vals, y_vals, grid = create_meshgrid_2d(features, bounds)
		Z_class = classifier.predict(grid).reshape(xx.shape)

		traces.insert(0, create_decision_regions_2d(
			xx, yy, x_vals, y_vals, Z_class, labels, label_to_color
		))

		traces.extend(
			create_decision_boundary_and_margins_2d(
				xx, yy, x_vals, y_vals, classifier, highlight_support
			)
		)

	layout = create_layout_2d(features, bounds)

	plotly.graph_objects.Figure(data=traces, layout=layout).show()

