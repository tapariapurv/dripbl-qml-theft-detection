"""128-d CNN latent -> 4-d quantum angles, as one fitted joblib-serializable object.

Standardize -> PCA -> keep the 4 most class-discriminative components -> robust
min-max scaling into [-angle_scale, angle_scale] (always inside [-pi, pi]).

Two details matter a great deal for downstream quantum-kernel accuracy:

* Component *selection*. Plain PCA keeps the highest-variance directions, which
  for a CNN latent are usually scene brightness/texture rather than the class
  signal — so a 128->4 bottleneck can throw the label information away. We take
  the PCA basis but keep the 4 components that separate the classes best
  (largest point-biserial correlation with the label).
* Angle *range*. A ZZFeatureMap is periodic in its inputs: spreading features
  over the full [-pi, pi] makes the kernel oscillate until the Gram matrix is
  nearly the identity (exponential concentration) and the SVM cannot generalize.
  `angle_scale` keeps the encoding in a smooth sub-range; it is tuned on a
  validation split at training time.
"""
from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAAngleProjector:
    def __init__(self, n_components: int = 4, pool_components: int = 32,
                 angle_scale: float = np.pi, mode: str = "pca") -> None:
        self.n_components = n_components
        self.pool_components = pool_components
        self.angle_scale = float(angle_scale)
        self.mode = mode  # "pca" (unsupervised) or "pls" (supervised projection)

    def fit(self, embeddings: np.ndarray, y: np.ndarray | None = None) -> "PCAAngleProjector":
        self.scaler = StandardScaler().fit(embeddings)
        if self.mode == "pls":
            # Supervised: PLS directions maximise covariance WITH THE LABEL, so the
            # 128->4 bottleneck keeps class signal instead of whatever happens to
            # have the most variance. This is the single biggest lever on how much
            # of the CNN's accuracy survives into the 4 quantum angles.
            if y is None:
                raise ValueError("mode='pls' needs labels")
            self.pls = PLSRegression(n_components=self.n_components, scale=False)
            self.pls.fit(self.scaler.transform(embeddings), np.asarray(y, dtype=np.float64))
            kept = self.pls.transform(self.scaler.transform(embeddings))
            self.lo = np.percentile(kept, 1, axis=0)
            self.hi = np.percentile(kept, 99, axis=0)
            return self
        pool = min(self.pool_components, *embeddings.shape)
        self.pca = PCA(n_components=pool, random_state=42).fit(self.scaler.transform(embeddings))

        projected = self.pca.transform(self.scaler.transform(embeddings))
        if y is None:
            self.components_kept = np.arange(self.n_components)
        else:  # point-biserial correlation of each component with the binary label
            y = np.asarray(y, dtype=np.float64)
            corr = np.abs(
                (projected - projected.mean(0)).T @ (y - y.mean())
                / (projected.std(0) * y.std() * len(y) + 1e-12)
            )
            self.components_kept = np.sort(np.argsort(corr)[::-1][:self.n_components])

        kept = projected[:, self.components_kept]
        self.lo = np.percentile(kept, 1, axis=0)   # robust range: ignore 1% tails
        self.hi = np.percentile(kept, 99, axis=0)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(embeddings)
        if getattr(self, "mode", "pca") == "pls":
            kept = self.pls.transform(scaled)
        else:
            kept = self.pca.transform(scaled)[:, self.components_kept]
        unit = (kept - self.lo) / np.maximum(self.hi - self.lo, 1e-12)  # ~[0, 1]
        angles = (2.0 * unit - 1.0) * self.angle_scale
        return np.clip(angles, -np.pi, np.pi).astype(np.float32)  # strict [-pi, pi]
