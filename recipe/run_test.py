import numpy as np
import robust_laplacian as rl


verts = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
faces = np.array(
    [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ],
    dtype=np.int64,
)

laplacian, mass = rl.mesh_laplacian(verts, faces)

assert laplacian.shape == (4, 4)
assert mass.shape == (4, 4)
assert laplacian.nnz > 0
assert mass.nnz == 4
assert np.allclose((laplacian - laplacian.T).data, 0.0, atol=1e-8)
assert np.all(mass.diagonal() > 0.0)
