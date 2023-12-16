#!/usr/bin/env python

"""Calibration routines."""

import glob
import cv2
assert int(cv2.__version__[0]) > 3, 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np


def calibrate_fisheye(images_path, CHECKERBOARD=(7, 5)):
    """Calibrate a camera with fisheye lenses"""

    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    corner_flags = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE

    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    imgfiles = []

    for fname in images_path:
        img = cv2.imread(fname)
        if _img_shape is None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, corner_flags)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)
            imgfiles.append(fname)

    img_size = (gray.shape[1], gray.shape[0])
    N_OK = len(objpoints)

    # ===================

    flags = None
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
        imgpoints, img_size, cameraMatrix=None, distCoeffs=None, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    # ====================

    # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
    #         cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    # rms, K, D, rvecs, tvecs = \
    #     cv2.fisheye.calibrate(
    #         objpoints,
    #         imgpoints,
    #         gray.shape[::-1],
    #         K=None,
    #         D=None,
    #         flags=calibration_flags,
    #         criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    #     )
        
    
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    print("rms=" + str(rms))

    # out = {'K':K, 'D':D, 'Rc':rvecs, 'tc':tvecs}
    return {'mtx':K, 'dist':D, 'rvecs':rvecs, 'tvecs':tvecs, 'imgfile':imgfiles, \
        'objpoints':objpoints, 'imgpoints':imgpoints, 'img_size':img_size, 'ret':rms}

if __name__ == '__main__':
    IMAGESPATH = glob.glob("images/img*.jpg")
    calibrate_fisheye(IMAGESPATH)
