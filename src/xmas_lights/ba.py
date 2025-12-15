import torch 
import numpy as np
from scipy.spatial.transform import Rotation as R

def mat_to_se_batch(T):
    r = R.from_matrix(T[...,:3, :3]).as_rotvec()
    t = T[...,:3, 3]
    return np.hstack([r, t])

def torch_least_squares(
    residual_fn,
    x0,
    args=(),
    max_iters=50,
    lr=1.0,
    loss='linear',
    huber_delta=1.0,
    verbose=True,
):
    """
    residual_fn: callable(x, *args) -> residual vector
    x0: (D,) initial torch tensor
    """

    x = x0.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [x],
        lr=lr,
        max_iter=20,
        line_search_fn='strong_wolfe'
    )

    def robust_loss(r):
        if loss == 'linear':
            return 0.5 * (r ** 2)
        elif loss == 'huber':
            abs_r = torch.abs(r)
            mask = abs_r < huber_delta
            return torch.where(
                mask,
                0.5 * r**2,
                huber_delta * (abs_r - 0.5 * huber_delta)
            )
        else:
            raise ValueError(f"Unknown loss: {loss}")

    prev_cost = None

    for it in range(max_iters):

        def closure():
            optimizer.zero_grad(set_to_none=True)
            r = residual_fn(x, *args)
            cost = robust_loss(r).mean()
            cost.backward()
            return cost

        cost = optimizer.step(closure)
        cost_val = cost.item()
        # simple convergence check
        if prev_cost is not None and abs(prev_cost - cost_val) < 1e-12:
            break
        
        if verbose and not (it % 10):
            if prev_cost is None:
                print(f"[{it:02d}] cost = {cost_val:.6f}")
            else:
                print(f"[{it:02d}] cost = {cost_val:.6f}  Δ={prev_cost - cost_val:.6f}")
            
        prev_cost = cost_val

    return x.detach()

def se3_to_mat_batch(x):
    """
    x: (N, 6) array
       x[:, :3] = rotvec (rx, ry, rz)
       x[:, 3:] = translation (tx, ty, tz)

    returns: (N, 4, 4) SE(3) matrices
    """

    N = x.shape[0]

    w = x[:, :3]     # rotation vectors
    t = x[:, 3:]     # translations

    theta = torch.linalg.norm(w, axis=1, keepdims=True)  # (N,1)
    theta2 = theta ** 2

    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    zeros = torch.zeros(N)

    W = torch.stack([
        zeros, -wz,    wy,
        wz,     zeros, -wx,
       -wy,     wx,    zeros
    ], dim=1).reshape(N, 3, 3)

    W2 = W @ W
    I = torch.eye(3)[None, :, :]

    eps = 1e-12

    A = torch.where(theta < eps,
                 1.0 - theta2 / 6.0,
                 torch.sin(theta) / theta)

    B = torch.where(theta < eps,
                 0.5 - theta2 / 24.0,
                 (1.0 - torch.cos(theta)) / theta2)

    A = A[:, None]   # (N,1,1)
    B = B[:, None]

    R = I + A * W + B * W2

    T = torch.zeros((N, 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0

    return T

def project_batch(K, T, Xs):
    """
    T: 4x4 (camera poses)
    Xs: Mx3
    """
    R, t = T[:3,:3], T[:3,3]
    Xc = Xs @ R.T + t      # world → camera
    x = Xc @ K.T
    return x[:, :2] / x[:, [2]]

def ba_residuals_batch(x, obs, cameraMatrix):
    num_views = obs.shape[0]
    cam_params = x[:6*(num_views-1)]
    pts  = x[6*(num_views-1):-1].reshape(-1, 3)

    K = cameraMatrix.clone()
    K[0,0] = x[-1]
    K[1,1] = x[-1]

    tfs = torch.concat([torch.eye(4, dtype=x.dtype)[None,:,:], 
                        se3_to_mat_batch(cam_params.reshape(-1, 6))], 0)
    obs_h = torch.stack([project_batch(K, tf, pts) for tf in tfs], 0)
    res = (obs_h - obs).reshape(-1)
    valid = ~torch.isnan(res)

    return res[valid], obs_h

