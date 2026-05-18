"""
procrustes_heatmap.py

Teacher-student trajectory alignment in the 14-D behavioural nullspace, with
joint sample-correspondence and orthogonal Procrustes optimisation.

Public API
----------
align_trajectories(X, Y, ...)              -> AlignmentResult
compute_heatmap(teacher_trajs, student_trajs, ...) -> dict
permutation_test(D, ...)                   -> dict

All trajectory arrays must have shape (N_trials, S_samples, T_timesteps, 14)
and must already be projected into the behavioural nullspace.  Use
project_to_nullspace() to convert 16-D hidden states.

GPU acceleration
----------------
Pass device='cuda' (or 'cuda:0', etc.) to align_trajectories / compute_heatmap.
Cost-matrix computation and Procrustes SVD run on the GPU; the Hungarian step
uses scipy on CPU (12 ms per 512×512 call, fast enough with thread parallelism).
The cross-term is computed via torch.bmm on (N, S, T*M) views to avoid the
O(N·S²·T·M) intermediate that a naive einsum would allocate.

Dependencies: numpy, scipy, joblib; torch optional (required for GPU path).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment

try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


NULLSPACE_DIM = 14


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    R: np.ndarray            # (14, 14) orthogonal rotation
    c: float                 # positive global scale factor
    matches: np.ndarray      # (N_trials, S_samples) int — per-trial sample permutation
    residual: float          # ||c·X@R − Y_matched||_F / ||X||_F  (best restart)
    objective_trace: list    # residual at each alternating-min iteration (best restart)
    restart_residuals: list  # final residual from each of the n_restarts
    identity_residual: float # residual with R=I and no sample remapping


# ---------------------------------------------------------------------------
# Projection utility
# ---------------------------------------------------------------------------

def project_to_nullspace(trajectories: np.ndarray, nullspace: np.ndarray) -> np.ndarray:
    """
    Project hidden-state trajectories from the 16-D ambient space into the
    14-D behavioural nullspace.

    Parameters
    ----------
    trajectories : (..., 16) array_like
    nullspace    : (14, 16) array_like — rows are orthonormal nullspace vectors
                   (model.behaviour_nullspace as a numpy array)

    Returns
    -------
    (..., 14) float64 ndarray
    """
    return np.asarray(trajectories, dtype=np.float64) @ np.asarray(nullspace, dtype=np.float64).T


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return Q.astype(np.float64)


def _compute_cost_matrices(
    X: np.ndarray, Y: np.ndarray, R: np.ndarray, c: float
) -> np.ndarray:
    """
    X   : (N, S, T, M)
    Y   : (N, S, T, M)
    R   : (M, M) current rotation
    c   : scalar  current scale

    Returns (N, S, S) squared Frobenius distances between c·X[n,k]@R and
    Y[n,j] for every trial n and sample pair (k, j).

    ||c·Xr[n,k] - Y[n,j]||_F^2
        = c^2·||X[n,k]||^2 + ||Y[n,j]||^2 - 2c·<X[n,k]@R, Y[n,j]>_F
    Expanding with einsum avoids an explicit (N, S, S, T, M) intermediate.
    """
    Xr = c * (X @ R)                                # (N, S, T, M)
    Xr_sq = (Xr ** 2).sum(axis=(-2, -1))            # (N, S)
    Y_sq  = (Y  ** 2).sum(axis=(-2, -1))            # (N, S)
    cross = np.einsum('nktm,njtm->nkj', Xr, Y)     # (N, S_k, S_j)
    return Xr_sq[:, :, None] + Y_sq[:, None, :] - 2.0 * cross   # (N, S, S)


def _lap_one_trial(cost_n: np.ndarray) -> np.ndarray:
    """Solve the assignment problem for one (S, S) cost matrix."""
    _, col_ind = linear_sum_assignment(cost_n)
    return col_ind.astype(np.int64)


def _gather_matched(
    X: np.ndarray, Y: np.ndarray, matches: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Advanced-index the matched student samples and pool over (N, S).

    X       : (N, S, T, M)
    Y       : (N, S, T, M)
    matches : (N, S) int — matches[n, k] = student sample index for teacher k

    Returns pooled_X, pooled_Y_matched, each (N*S, T, M).
    """
    N, S, T, M = X.shape
    Y_matched = Y[np.arange(N)[:, None], matches]   # (N, S, T, M)
    return X.reshape(N * S, T, M), Y_matched.reshape(N * S, T, M)


def _solve_procrustes(
    pooled_X: np.ndarray, pooled_Y: np.ndarray, allow_scaling: bool
) -> tuple[np.ndarray, float]:
    """
    Closed-form scaled orthogonal Procrustes:
        minimise  ||c · X @ R - Y||_F^2

    SVD of cross-covariance A = Xf.T @ Yf:
        R = U @ Vt
        c = trace(R.T @ A) / ||X||_F^2  =  sum(singular values) / ||X||_F^2

    c is fixed at 1.0 when allow_scaling is False.
    """
    M = pooled_X.shape[-1]
    Xf = pooled_X.reshape(-1, M)
    Yf = pooled_Y.reshape(-1, M)
    A = Xf.T @ Yf
    U, s, Vt = np.linalg.svd(A)
    R = U @ Vt
    if allow_scaling:
        c = float(s.sum()) / (float((Xf ** 2).sum()) + 1e-30)
        c = max(c, 1e-10)
    else:
        c = 1.0
    return R, c


def _normalised_residual(
    pooled_X: np.ndarray, pooled_Y: np.ndarray, R: np.ndarray, c: float
) -> float:
    """||c · X @ R - Y||_F / ||X||_F"""
    diff = c * (pooled_X @ R) - pooled_Y
    return float(
        np.sqrt((diff ** 2).sum()) / (np.sqrt((pooled_X ** 2).sum()) + 1e-30)
    )


def _identity_residual(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Residual with R = I and the identity sample permutation (teacher sample k
    paired with student sample k in the same trial, in natural order).
    Used as a sanity-check baseline: measures how much the alignment actually helps.
    """
    N, S, T, M = X.shape
    diff = X.reshape(N * S, T, M) - Y.reshape(N * S, T, M)
    Xf   = X.reshape(N * S, T, M)
    return float(
        np.sqrt((diff ** 2).sum()) / (np.sqrt((Xf ** 2).sum()) + 1e-30)
    )


def _run_one_restart(
    X: np.ndarray,
    Y: np.ndarray,
    R_init: np.ndarray,
    allow_scaling: bool,
    max_iter: int,
    tol: float,
    n_jobs: int,
) -> tuple:
    """
    One alternating-minimisation run from a single (R_init, c=1) initialisation.
    Returns (R, c, matches, final_residual, objective_trace).
    """
    N = X.shape[0]
    R, c, prev_obj = R_init.copy(), 1.0, np.inf
    trace = []

    for _ in range(max_iter):
        # A-step: solve LAP per trial, in parallel with threads (LAP releases GIL)
        cost = _compute_cost_matrices(X, Y, R, c)          # (N, S, S)
        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_lap_one_trial)(cost[n]) for n in range(N)
        )
        matches = np.stack(results, axis=0)                # (N, S)

        # B-step: closed-form Procrustes on all matched pairs pooled
        pooled_X, pooled_Y = _gather_matched(X, Y, matches)
        R, c = _solve_procrustes(pooled_X, pooled_Y, allow_scaling)

        obj = _normalised_residual(pooled_X, pooled_Y, R, c)
        trace.append(float(obj))

        if abs(prev_obj - obj) / (abs(prev_obj) + 1e-30) < tol:
            break
        prev_obj = obj

    return R, c, matches, float(obj), trace


# ---------------------------------------------------------------------------
# GPU helpers (torch required)
# ---------------------------------------------------------------------------

def _compute_cost_matrices_gpu(
    X: '_torch.Tensor', Y: '_torch.Tensor', R: '_torch.Tensor', c: float
) -> '_torch.Tensor':
    """
    GPU cost matrix via torch.bmm.  Avoids the O(N·S²·T·M) intermediate
    that a naive einsum would create — instead reshapes to (N, S, T*M) and
    uses batched matmul for the cross term.

    X, Y : (N, S, T, M) float32 tensors on device
    R    : (M, M) float32 tensor on device
    Returns (N, S, S) cost tensor on device.
    """
    N, S, T, M = X.shape
    Xr = c * _torch.matmul(X, R)              # (N, S, T, M)
    Xr_sq = (Xr ** 2).sum(dim=(-2, -1))       # (N, S)
    Y_sq  = (Y  ** 2).sum(dim=(-2, -1))       # (N, S)
    # Reshape to (N, S, T*M) and use batched matmul for the (N, S, S) cross term.
    cross = _torch.bmm(
        Xr.reshape(N, S, T * M),
        Y.reshape(N, S, T * M).transpose(1, 2),
    )                                          # (N, S, S)
    return Xr_sq[:, :, None] + Y_sq[:, None, :] - 2.0 * cross


def _gather_matched_gpu(
    X: '_torch.Tensor', Y: '_torch.Tensor', matches: np.ndarray
) -> 'tuple[_torch.Tensor, _torch.Tensor]':
    N, S, T, M = X.shape
    matches_t = _torch.as_tensor(matches, dtype=_torch.long, device=X.device)
    Y_matched = Y[_torch.arange(N, device=X.device)[:, None], matches_t]
    return X.reshape(N * S, T, M), Y_matched.reshape(N * S, T, M)


def _solve_procrustes_gpu(
    pooled_X: '_torch.Tensor', pooled_Y: '_torch.Tensor', allow_scaling: bool
) -> 'tuple[_torch.Tensor, float]':
    M = pooled_X.shape[-1]
    Xf = pooled_X.reshape(-1, M)
    Yf = pooled_Y.reshape(-1, M)
    U, s, Vh = _torch.linalg.svd(Xf.T @ Yf)
    R = U @ Vh
    if allow_scaling:
        c = float(s.sum().item()) / (float((Xf ** 2).sum().item()) + 1e-30)
        c = max(c, 1e-10)
    else:
        c = 1.0
    return R, c


def _normalised_residual_gpu(
    pooled_X: '_torch.Tensor', pooled_Y: '_torch.Tensor',
    R: '_torch.Tensor', c: float
) -> float:
    diff = c * _torch.matmul(pooled_X, R) - pooled_Y
    return float(
        diff.pow(2).sum().sqrt().item()
        / (pooled_X.pow(2).sum().sqrt().item() + 1e-30)
    )


def _identity_residual_gpu(X: '_torch.Tensor', Y: '_torch.Tensor') -> float:
    N, S, T, M = X.shape
    Xf   = X.reshape(N * S, T, M)
    diff = Xf - Y.reshape(N * S, T, M)
    return float(
        diff.pow(2).sum().sqrt().item()
        / (Xf.pow(2).sum().sqrt().item() + 1e-30)
    )


def _run_one_restart_gpu(
    X_t: '_torch.Tensor',
    Y_t: '_torch.Tensor',
    R_init: np.ndarray,
    allow_scaling: bool,
    max_iter: int,
    tol: float,
    n_jobs: int,
) -> tuple:
    """
    One alternating-min restart on GPU.
    Cost matrix and SVD on device; Hungarian on CPU via scipy threads.
    Returns (R_np, c, matches, final_residual, objective_trace) — R as float64 numpy.
    """
    N = X_t.shape[0]
    R_t = _torch.as_tensor(R_init, dtype=X_t.dtype, device=X_t.device)
    c, prev_obj = 1.0, float('inf')
    trace: list = []
    matches: np.ndarray | None = None

    for _ in range(max_iter):
        # A-step: cost on GPU → CPU for parallel scipy LAP
        cost_np = _compute_cost_matrices_gpu(X_t, Y_t, R_t, c).cpu().numpy().astype(np.float64)
        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_lap_one_trial)(cost_np[n]) for n in range(N)
        )
        matches = np.stack(results, axis=0)

        # B-step: gather + Procrustes on GPU
        pooled_X, pooled_Y = _gather_matched_gpu(X_t, Y_t, matches)
        R_t, c = _solve_procrustes_gpu(pooled_X, pooled_Y, allow_scaling)

        obj = _normalised_residual_gpu(pooled_X, pooled_Y, R_t, c)
        trace.append(obj)
        if abs(prev_obj - obj) / (abs(prev_obj) + 1e-30) < tol:
            break
        prev_obj = obj

    return R_t.cpu().numpy().astype(np.float64), c, matches, obj, trace


# ---------------------------------------------------------------------------
# Public function 1: single-pair alignment
# ---------------------------------------------------------------------------

def align_trajectories(
    X: np.ndarray,
    Y: np.ndarray,
    allow_scaling: bool = True,
    n_restarts: int = 3,
    max_iter: int = 50,
    tol: float = 1e-5,
    n_jobs: int = -1,
    seed: int = 42,
    device: str = 'cpu',
) -> AlignmentResult:
    """
    Align teacher trajectories X to student trajectories Y in the 14-D
    behavioural nullspace via joint sample-correspondence and orthogonal
    Procrustes optimisation.

    The optimisation alternates between:
      A-step: fix (R, c), solve a Hungarian assignment per trial on the
              pairwise squared Frobenius cost matrix (batched via bmm).
      B-step: fix the assignment, solve scaled orthogonal Procrustes in
              closed form via SVD of the pooled cross-covariance.

    Parameters
    ----------
    X, Y         : (N_trials, S_samples, T_timesteps, 14) float arrays,
                   already projected into the 14-D behavioural nullspace.
    allow_scaling: optimise global scale c > 0 alongside rotation R.
                   Set False for pure orthogonal Procrustes (c = 1).
    n_restarts   : number of initialisations; restart 0 always uses R = I.
    max_iter     : max alternating-minimisation iterations per restart.
    tol          : convergence: relative change in residual below this value.
    n_jobs       : joblib thread-pool size for the per-trial Hungarian step.
                   -1 uses all available CPUs.
    seed         : RNG seed for random orthogonal initialisations.
    device       : 'cpu' (default) or a torch device string ('cuda', 'cuda:0',
                   etc.).  GPU path runs cost-matrix computation and Procrustes
                   SVD on the device; Hungarian stays on CPU.

    Returns
    -------
    AlignmentResult with the best result across all restarts.
    """
    assert X.ndim == 4 and Y.ndim == 4, "Trajectories must be (N, S, T, M)"
    assert X.shape == Y.shape, f"Shape mismatch: X={X.shape}, Y={Y.shape}"
    assert X.shape[3] == NULLSPACE_DIM, (
        f"Last dim must be {NULLSPACE_DIM} (nullspace); got {X.shape[3]}"
    )

    use_gpu = (device != 'cpu') and _HAS_TORCH
    M = NULLSPACE_DIM
    rng = np.random.default_rng(seed)

    if use_gpu:
        X_t = _torch.as_tensor(np.asarray(X, dtype=np.float32)).to(device)
        Y_t = _torch.as_tensor(np.asarray(Y, dtype=np.float32)).to(device)
        # Remove the temporal mean of every trajectory — makes alignment
        # invariant to translation (absolute position in nullspace).
        # Equivalent to the centering in compare_two_models_prep_trajectories.
        X_t = X_t - X_t.mean(dim=2, keepdim=True)   # (N, S, T, M)
        Y_t = Y_t - Y_t.mean(dim=2, keepdim=True)
        id_res = _identity_residual_gpu(X_t, Y_t)
    else:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        X = X - X.mean(axis=2, keepdims=True)        # (N, S, T, M)
        Y = Y - Y.mean(axis=2, keepdims=True)
        id_res = _identity_residual(X, Y)

    best: dict = dict(R=None, c=1.0, matches=None, residual=np.inf, trace=[])
    restart_residuals = []

    for restart_idx in range(n_restarts):
        R_init = np.eye(M) if restart_idx == 0 else _random_orthogonal(M, rng)
        if use_gpu:
            R, c, matches, residual, trace = _run_one_restart_gpu(
                X_t, Y_t, R_init, allow_scaling, max_iter, tol, n_jobs
            )
        else:
            R, c, matches, residual, trace = _run_one_restart(
                X, Y, R_init, allow_scaling, max_iter, tol, n_jobs
            )
        restart_residuals.append(float(residual))
        if residual < best['residual']:
            best.update(R=R, c=c, matches=matches, residual=residual, trace=trace)

    return AlignmentResult(
        R=best['R'],
        c=float(best['c']),
        matches=best['matches'],
        residual=float(best['residual']),
        objective_trace=best['trace'],
        restart_residuals=restart_residuals,
        identity_residual=float(id_res),
    )


# ---------------------------------------------------------------------------
# Public function 2: pairwise heatmap
# ---------------------------------------------------------------------------

def compute_heatmap(
    teacher_trajs: list,
    student_trajs: list,
    allow_scaling: bool = True,
    n_restarts: int = 3,
    max_iter: int = 50,
    tol: float = 1e-5,
    n_jobs_lap: int = -1,
    seed: int = 42,
    device: str = 'cpu',
) -> dict:
    """
    Compute alignment-derived Procrustes residuals for all teacher × student pairs.

    Each cell is independently optimised (its own R, c, and sample matching).
    This is intentional: the heatmap measures shape difference after removing
    all nuisance degrees of freedom, so per-cell optimisation is correct.

    Parameters
    ----------
    teacher_trajs : list of n_teachers arrays, each (N, S, T, 14)
    student_trajs : list of n_students  arrays, each (N, S, T, 14)
    allow_scaling : passed through to align_trajectories for every cell
    device        : torch device string; 'cpu' or 'cuda' / 'cuda:0' etc.
    [other args]  : passed through to align_trajectories

    Returns
    -------
    dict with:
        'residuals'         : (n_teachers, n_students) float64 ndarray
        'alignment_results' : list-of-lists of AlignmentResult, indexed [i][j]
    """
    n_teachers = len(teacher_trajs)
    n_students  = len(student_trajs)
    residuals   = np.full((n_teachers, n_students), np.nan, dtype=np.float64)
    all_results = [[None] * n_students for _ in range(n_teachers)]

    for i, X in enumerate(teacher_trajs):
        for j, Y in enumerate(student_trajs):
            result = align_trajectories(
                X, Y,
                allow_scaling=allow_scaling,
                n_restarts=n_restarts,
                max_iter=max_iter,
                tol=tol,
                n_jobs=n_jobs_lap,
                seed=seed + i * n_students + j,
                device=device,
            )
            residuals[i, j] = result.residual
            all_results[i][j] = result
            print(
                f"  [{i:2d},{j:2d}]  residual={result.residual:.4f}  "
                f"id_residual={result.identity_residual:.4f}  "
                f"c={result.c:.3f}  "
                f"restarts={[f'{r:.4f}' for r in result.restart_residuals]}"
            )

    return {'residuals': residuals, 'alignment_results': all_results}


# ---------------------------------------------------------------------------
# Public function 3: permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    D: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Permutation test for diagonal structure in the residual heatmap.

    Test statistic: T = mean(diagonal) - mean(off-diagonal).
    Smaller T means same-index teacher-student pairs align better than
    cross-index pairs (specificity).

    The null permutes which student column counts as the 'diagonal' for each
    row, keeping the matrix values themselves fixed.  Only n numbers per
    permutation need to be summed, making 10 k permutations very cheap.

    Parameters
    ----------
    D              : (n, n) residual matrix (square)
    n_permutations : size of the Monte Carlo null distribution
    seed           : RNG seed

    Returns
    -------
    dict with keys:
        'observed_T'       : float
        'null_distribution': (n_permutations,) float64 ndarray
        'p_value'          : float — one-sided P(null T ≤ observed T)
        'diag_mean'        : float
        'off_diag_mean'    : float
    """
    assert D.ndim == 2 and D.shape[0] == D.shape[1], "D must be square"
    n = D.shape[0]
    rng = np.random.default_rng(seed)

    total_sum = float(D.sum())
    n_off = n * (n - 1)

    diag_sum_obs     = float(np.diag(D).sum())
    diag_mean_obs    = diag_sum_obs / n
    off_diag_mean_obs = (total_sum - diag_sum_obs) / n_off
    observed_T = diag_mean_obs - off_diag_mean_obs

    rows   = np.arange(n)
    null_T = np.empty(n_permutations, dtype=np.float64)
    for k in range(n_permutations):
        perm = rng.permutation(n)
        ds   = float(D[rows, perm].sum())
        null_T[k] = ds / n - (total_sum - ds) / n_off

    p_value = float((null_T <= observed_T).mean())

    return {
        'observed_T':        float(observed_T),
        'null_distribution': null_T,
        'p_value':           p_value,
        'diag_mean':         float(diag_mean_obs),
        'off_diag_mean':     float(off_diag_mean_obs),
    }
