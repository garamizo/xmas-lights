import cv2 
import torch 
from pathlib import Path 
from typing import List, Tuple 
import numpy as np


def read_video(path: Path) -> List[np.ndarray]:
    # preload video at downsampled rate
    cap = cv2.VideoCapture(path)

    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return frames


def select_frames(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
    '''
    Docstring for select_frames
    
    :param frames: Description
    :type frames: List[np.ndarray]
    :return: Description
    :rtype: List[ndarray]
    '''
    num_frames = len(frames)
    bright = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean() for im in frames]
    vlims = np.percentile(bright[-num_frames*2//3:], [5, 95])
    thresh = np.mean(vlims)
    bright_norm = (np.array(bright) - thresh) * 2 / np.diff(vlims)

    idx_up = np.argwhere((bright_norm[1:] > 0) & (bright_norm[:-1] < 0)).flatten() + 1
    # return None, bright_norm, None
    assert len(idx_up) >= 2, "Could not find brightness transitions" 
    idx_up = idx_up[-2:]
    
    return frames[idx_up[0]:idx_up[1]], bright_norm[idx_up[0]:idx_up[1]], bright_norm

def init_camera_matrix(frame):
    h, w, _ = frame.shape
    f = 0.673 * h
    return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], np.float32)


def filter_frames(frames: List[np.ndarray], num_flashes: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Average franes corresponding to the same flashing state
    Calculate foreground
    
    :param frames: Frame sequence of videos, properly segmented
    :param num_flashes: Number of flashes (digits)
    """
    
    frames_np = np.array(frames)
    # len_batch * (num_flashes+1) == num_frames
    len_batch = int(np.round(len(frames) / (num_flashes + 1)))
    frames_filt = []
    for i in range(num_flashes+1):
        rng = range(i*len_batch + 2, (i+1)*len_batch - 2)
        frames_filt.append(np.mean(frames_np[rng], axis=0).astype(np.uint8))

    # TODO Should off state frame (last) be included here?
    im_std = np.max(np.std(frames_filt, axis=0), axis=-1)
    thresh = np.mean(np.percentile(im_std, [50, 100]))
    fg = im_std > thresh 

    # diag = (frames[0].shape[0]**2 + frames[0].shape[1]**2)**0.5
    # ks = int((diag // 300) // 2 * 2 + 1)
    # frames_filt = [cv2.GaussianBlur(f, (ks, ks), 0) for f in frames_filt]

    return frames_filt[:-1], fg


def digits_from_frame(frame, fg, num_hues):
    """
    Docstring for digits_from_frame
    
    :param frame: Description
    :param fg: Description
    :param num_hues: Description
    """
    im = frame * fg[:,:,None]
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV) #[180,255,255]
    hue = hsv[:,:,0].astype(float) * 2  # to 0-360
    digit = np.round(((hue + 180/num_hues) % 360) * num_hues / 360 - 0.5)

    return (digit * fg).astype(np.uint8)


def deriv(x):
    if len(x.shape) == 1:
        x = x[:,None]
    return np.r_[x[[1]] - x[[0]],
                 (x[2:] - x[:-2]) / 2,
                 x[[-1]] - x[[-2]]]


def xy_from_image(im_id, fg):
    diag_im = (im_id.shape[0]**2 + im_id.shape[1]**2)**0.5

    ksz = int(diag_im / 500)
    ksz = ksz // 2 * 2 + 1
    kernel = np.ones((ksz, ksz), np.uint8)
    AREA_MIN = diag_im * diag_im * 1e-6
    DXY_MAX = diag_im / 20

    xys, areas, ids = [], [], []
    for i in np.unique(im_id[fg]):
        im_i = ((im_id == i) & fg).astype(np.uint8)    
        opening = cv2.morphologyEx(im_i, cv2.MORPH_CLOSE, kernel)

        _, _, stats, centroids = cv2.connectedComponentsWithStats(
            opening, connectivity=4, ltype=cv2.CV_32S)

        idx = np.argmax(stats[1:,-1]) + 1
        left, top, width, height, area = stats[idx]
        if area > AREA_MIN:
            xys.append(centroids[idx])
            areas.append(stats[idx,4])
            ids.append(i)

    xys, ids = np.asarray(xys), np.asarray(ids)
    dxy1 = np.diff(xys, axis=0)[1:] / np.diff(ids)[1:,None]
    dxy2 = np.diff(xys, axis=0)[:-1] / np.diff(ids)[:-1,None]
    inliers = np.r_[True, 
        (np.sum(dxy1**2, 1)**0.5 < DXY_MAX) | (np.sum(dxy2**2, 1)**0.5 < DXY_MAX),
        True]

    return xys[inliers], ids[inliers], np.asarray(areas)[inliers]


def filt_sequence_inliers(pts, ids=None):
    if ids is None:
        ids = np.arange(pts.shape[0])
    std_pts = np.diff(np.quantile(pts - np.median(pts,0), [0.2, 0.8]))
    std_ids = np.diff(np.quantile(ids, [0.2, 0.8]))
    DS_MAX = 200 * std_pts / std_ids

    ds = np.sum(np.diff(pts, axis=0)**2, 1)**0.5 / np.diff(ids) / DS_MAX
    inliers =  np.r_[True, 
        (ds[1:] < 1) | (ds[:-1] < 1),
        True]
    pts_filt = pts.copy()
    pts_filt[~inliers] = np.array([np.interp(ids[~inliers], ids[inliers], ptdim[inliers]) 
                                for ptdim in pts_filt.T]).T
    return pts_filt, inliers


def init_camera_pose(pts_xy: List[np.ndarray], cameraMatrix: np.ndarray):
    """
    Docstring for init_camera_pose
    Returns edges dict with
        i:
        j:
        T_ij:
        intersect_count:
    
    :param pts_xy: Description
    :type pts_xy: List[np.ndarray]
    :param cameraMatrix: Description
    :type cameraMatrix: np.ndarray
    """
    num_views = len(pts_xy)

    edges = []   # pose-graph edges
    # assume first view camera is origin, compute pose of other views in respect to first
    for i in range(num_views):
        for j in range(num_views):#range(i+1, num_views):
            # ids, idxi, idxj = np.intersect1d(caps[i]['ids'], caps[j]['ids'], 
            #                                 return_indices=True)
            ptsi, ptsj = pts_xy[i], pts_xy[j]
            ids_rows = ~(np.any(np.isnan(ptsi), 1) | np.any(np.isnan(ptsj), 1))
            ptsi, ptsj = ptsi[ids_rows], ptsj[ids_rows]

            # assert np.all(caps[i]['ids'][idxi] == caps[j]['ids'][idxj]), "oops"

            E, mask = cv2.findEssentialMat(ptsi, ptsj, cameraMatrix=cameraMatrix, 
                                method=cv2.RANSAC, prob=0.995, threshold=3.0)
            ptsi, ptsj = ptsi[mask.ravel() == 1], ptsj[mask.ravel() == 1]
            assert (E is not None) & (E.shape[0] == 3), "No Essential solution"
            _, R, t, _ = cv2.recoverPose(E, ptsi, ptsj, cameraMatrix=cameraMatrix)

            # Normalize scale assuming target has std=1
            P1 = cameraMatrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = cameraMatrix @ np.hstack((R, t))

            # Triangulate
            X_h = cv2.triangulatePoints(P1, P2, ptsi.T, ptsj.T)
            X = (X_h[:3] / X_h[3]).T
            scale = np.std(X - X.mean(axis=0))

            # get tf0j
            edges.append({"i": i, 
                          "j": j, 
                          "T_ij": np.r_[np.c_[R, t/scale], [[0, 0, 0, 1]]], 
                          "intersect_count": int(sum(mask.flatten())),
                          })

    return edges


def solve_pose_graph(edges, method="direct"):
    
    if method == "direct":
        nodes = np.unique([e["i"] for e in edges] + [e["j"] for e in edges])
        num_views = len(nodes)
        tfs = np.ones((num_views, 4, 4)) * np.nan
        tfs[0] = np.eye(4)
        for e in edges:
            if e['i'] == 0:
                tfs[e['j']] = e['T_ij']

        assert ~np.isnan(tfs).any(), "Graph could not be completed"
        return tfs
    else:
        raise ValueError(f"Methods available: {", ".join(['direct'])}")
           

def init_world_positions(pts_xy: List[np.ndarray], tfs: List[np.ndarray], cameraMatrix: np.ndarray):
    pts_dict = []#np.ones((num_views, num_views, num_bulbs, 3)) * np.nan
    num_views = len(pts_xy)
    num_bulbs = pts_xy[0].shape[0]

    for i in range(num_views):
        for j in range(num_views):#range(i+1, num_views):

            ptsi, ptsj = pts_xy[i], pts_xy[j]
            ids_rows = ~(np.any(np.isnan(ptsi), 1) | np.any(np.isnan(ptsj), 1))
            ptsi, ptsj = ptsi[ids_rows], ptsj[ids_rows]
                
            P1 = cameraMatrix @ tfs[i,:3,:]
            P2 = cameraMatrix @ tfs[j,:3,:]

            # Triangulate
            X_h = cv2.triangulatePoints(P1, P2, ptsi.T, ptsj.T)

            pts_dict.append(np.ones((num_bulbs, 3)) * np.nan)
            pts_dict[-1][ids_rows] = (X_h[:3] / X_h[3]).T

    pts = np.nanmedian(pts_dict, axis=0)
    pts_filt = pts.copy()
    pts_filt[np.any(np.isnan(pts), 1)] = np.nanmedian(pts, 0)
    pts_filt, _ = filt_sequence_inliers(pts_filt, np.arange(num_bulbs))
    return pts_filt