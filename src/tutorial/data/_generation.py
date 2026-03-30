import numpy as np
import pandas as pd


def generate_separable_dataset(
	n_points=300,
	spread=0.25,
	slope=0.5,
	intercept=0.0,
	n_blobs=13,
	radius=1.5,
	seed=42
):
	random = np.random.default_rng(seed)

	# generate blob centers
	centers = np.column_stack((
		random.uniform(-3, 3, n_blobs),
		random.uniform(-1, 1, n_blobs)
	))

	points_per_blob = n_points // n_blobs
	
	angles = random.uniform(0, 2*np.pi, (n_blobs, points_per_blob))
	radii = radius * np.sqrt(random.uniform(0, 1, (n_blobs, points_per_blob)))
	dx = radii * np.cos(angles)
	dy = radii * np.sin(angles)

	X = centers[:, None, :] + np.stack((dx, dy), axis=-1)
	X = X.reshape(-1, 2)

	# linear boundary: y = slope * x + intercept
	norm_vec = np.array([-slope, 1])
	norm_vec = norm_vec / np.linalg.norm(norm_vec)
	d = intercept / np.sqrt(slope**2 + 1)

	# compute signed distance and class
	dist = np.dot(X, norm_vec) - d
	y = (dist > 0).astype(int)

	# push points away from line
	X += spread * np.sign(dist)[:, np.newaxis] * norm_vec

	df = pd.DataFrame(X, columns=["x", "y"])
	df["target"] = y
	return df

def generate_non_separable_dataset(
	n_points=300,
	inner_radius=1.0,
	ring_width=1.0,
	spread=0.25,
	frac_pos=0.5,
	seed=42
):
	random = np.random.default_rng(seed)

	n_pos = int(n_points * frac_pos)
	n_neg = n_points - n_pos

	# inner circle
	theta_in = random.uniform(0, 2*np.pi, n_pos)
	r_in = np.sqrt(random.uniform(0, inner_radius, n_pos))
	X_in = np.column_stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)])

	# outer ring
	theta_out = random.uniform(0, 2*np.pi, n_neg)
	r_out = inner_radius + spread + random.uniform(0, ring_width, n_neg)
	X_out = np.column_stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)])

	X = np.vstack([X_in, X_out])
	y = np.hstack([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])

	df = pd.DataFrame(X, columns=["x", "y"])
	df["target"] = y
	return df

