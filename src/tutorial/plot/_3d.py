# src/tutorial/plot/_3d.py

import numpy as np
import pandas as pd
import plotly.graph_objects
from ._utils import extend_range, compute_plot_bounds, get_color_map


# ======================
# 3D HELPERS
# ======================

def create_meshgrid_3d(features, bounds, resolution=30):
	x_min, x_max = bounds[features[0]]
	y_min, y_max = bounds[features[1]]
	z_min, z_max = bounds[features[2]]

	x_vals = np.linspace(x_min, x_max, resolution)
	y_vals = np.linspace(y_min, y_max, resolution)
	z_vals = np.linspace(z_min, z_max, resolution)

	xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals)

	grid = pd.DataFrame({
		features[0]: xx.ravel(),
		features[1]: yy.ravel(),
		features[2]: zz.ravel()
	})

	return xx, yy, zz, grid


def create_class_traces_3d(X, y, features, labels, label_to_color):
	traces = []
	for label in labels:
		mask = y == label
		traces.append(plotly.graph_objects.Scatter3d(
			x=X[mask][features[0]],
			y=X[mask][features[1]],
			z=X[mask][features[2]],
			mode="markers",
			marker=dict(size=4, color=label_to_color[label]),
			name=f"Class {label}"
		))
	return traces


def create_decision_surfaces_3d(xx, yy, zz, Z, labels, label_to_color):
	traces = []

	if len(labels) != 2:
		raise ValueError("Binary classification only.")

	neg_label, pos_label = sorted(labels)
	neg_color = label_to_color[neg_label]
	pos_color = label_to_color[pos_label]

	def constant_colorscale(color):
		return [[0, color], [1, color]]

	def surface(level, color, name, opacity):
		return plotly.graph_objects.Isosurface(
			x=xx.ravel(),
			y=yy.ravel(),
			z=zz.ravel(),
			value=Z.ravel(),
			isomin=level,
			isomax=level,
			surface_count=1,
			opacity=opacity,
			showscale=False,
			colorscale=constant_colorscale(color),
			cmin=level,
			cmax=level,
			caps=dict(x_show=False, y_show=False, z_show=False),
			name=name
		)

	# Decision boundary (darker)
	traces.append(surface(0, "black", "Decision Boundary", opacity=0.7))

	# Margins colored by class
	traces.append(surface(+1, pos_color, "Margin +1", opacity=0.3))
	traces.append(surface(-1, neg_color, "Margin -1", opacity=0.3))

	return traces


def create_support_vectors_trace_3d(classifier):
	if not hasattr(classifier, "support_vectors_"):
		return None

	sv = classifier.support_vectors_

	return plotly.graph_objects.Scatter3d(
		x=sv[:, 0],
		y=sv[:, 1],
		z=sv[:, 2],
		mode="markers",
		marker=dict(
			size=8,
			color="black",
			symbol="circle-open",
			line=dict(width=3)
		),
		name="Support Vectors"
	)


def create_layout_3d(features, bounds):
	x_min, x_max = bounds[features[0]]
	y_min, y_max = bounds[features[1]]
	z_min, z_max = bounds[features[2]]

	axis_style = dict(
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
		backgroundcolor="white",
		showspikes=False  # disables hover spike lines
	)

	return plotly.graph_objects.Layout(
		scene=dict(
			xaxis=dict(title=features[0], range=[x_min, x_max], **axis_style),
			yaxis=dict(title=features[1], range=[y_min, y_max], **axis_style),
			zaxis=dict(title=features[2], range=[z_min, z_max], **axis_style),
			aspectmode="manual",
			aspectratio=dict(x=1, y=1, z=1),
			camera=dict(
				eye=dict(x=1.5, y=1.5, z=0.25)  # increase z to "look from above"
			)
		),
		hovermode=False,
		dragmode="turntable",
		margin=dict(l=0, r=0, t=40, b=0),
		paper_bgcolor="white"
	)


# ======================
# MAIN ENTRY POINT
# ======================
def plot_data_3d(df, features, target, classifier=None, highlight_support=True):
	if len(features) != 3:
		raise ValueError("plot_data_3d requires exactly 3 features")

	X = df[features]
	y = df[target].values
	labels = np.unique(y)

	bounds = compute_plot_bounds(X, features)
	label_to_color = get_color_map(labels, plotly.colors.qualitative.Plotly)

	traces = create_class_traces_3d(X, y, features, labels, label_to_color)

	if classifier is not None and hasattr(classifier, "decision_function"):
		xx, yy, zz, grid = create_meshgrid_3d(features, bounds)
		Z = classifier.decision_function(grid).reshape(xx.shape)

		traces.extend(
			create_decision_surfaces_3d(xx, yy, zz, Z, labels, label_to_color)
		)

	if classifier is not None and highlight_support:
		sv_trace = create_support_vectors_trace_3d(classifier)
		if sv_trace is not None:
			traces.append(sv_trace)

	layout = create_layout_3d(features, bounds)

	plotly.graph_objects.Figure(data=traces, layout=layout).show()

