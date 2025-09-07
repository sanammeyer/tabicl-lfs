"""Quick sanity check for TabICL installation.

Runs a very small inference using the scikit-learn wrapper on the Iris dataset.
The first run will download the pretrained checkpoint (may take a moment).

Usage (from repo root):
	python test.py
"""

from __future__ import annotations

import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from tabicl import TabICLClassifier


def main():
	X, y = load_iris(return_X_y=True, as_frame=True)  # returns DataFrame to test categorical handling path

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.3, random_state=0, stratify=y
	)

	# Keep it tiny so it runs fast; larger ensembles improve accuracy but are slower.
	clf = TabICLClassifier(
		n_estimators=2,          # minimal ensemble for speed
		batch_size=2,            # small batch
		use_amp=True,            # mixed precision if supported
		verbose=False,
		# leave other params default (will auto-download model if not cached)
	)

	print("Fitting (this only prepares transforms + downloads model; no gradient training)...")
	t0 = time.time()
	clf.fit(X_train, y_train)
	fit_time = time.time() - t0
	print(f"Fit done in {fit_time:.2f}s")

	print("Predicting...")
	t1 = time.time()
	y_pred = clf.predict(X_test)
	pred_time = time.time() - t1
	acc = accuracy_score(y_test, y_pred)
	print(f"Accuracy: {acc:.3f}  (prediction time {pred_time:.2f}s)")
	print("Sample predictions (first 5):")
	for i, (true_label, pred_label) in enumerate(zip(y_test.iloc[:5], y_pred[:5])):
		print(f"  #{i}: true={true_label}, pred={pred_label}")

	print("\nClassification report:")
	print(classification_report(y_test, y_pred, digits=3))

	print("Sanity check completed successfully.")


if __name__ == "__main__":
	main()

