#!/usr/bin/env python3

"""Localize bulbs from a Christmas tree."""

# import pickle
import numpy as np
# import matplotlib.pyplot as plt
import cv2
# from scipy.optimize import minimize, Bounds, shgo
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.signal import filtfilt


def draw_axis(img, axis_img):
    # draw origin from camera viewpoint
    img = cv2.line(img, tuple(axis_img[0,:,:].ravel()), tuple(axis_img[1,:,:].ravel()), (255,0,0), 10)
    img = cv2.line(img, tuple(axis_img[0,:,:].ravel()), tuple(axis_img[2,:,:].ravel()), (0,255,0), 10)
    img = cv2.line(img, tuple(axis_img[0,:,:].ravel()), tuple(axis_img[3,:,:].ravel()), (0,0,255), 10)
    return img


def Rx(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)

def Ry(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)

def Rz(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)

def Rod_from_Rxyz(rx, ry, rz):
    return cv2.Rodrigues(np.matmul(np.matmul(Rx(rx), Ry(ry)), Rz(rz)))[0]


def NumericalGradient(fun, x0, eps=1e-5):
    x0 = np.array(x0)
    N = len(x0)
    grad = np.zeros(N)
    for i in range(N):
        x = x0
        x[i] += eps
        f1 = fun(x)
        x[i] -= 2 * eps
        f2 = fun(x)
        grad[i] = (f1 - f2) / (2*eps)
    return(grad)


def print_c_matrix(a):
    msg = "{"
    for i in range(a.shape[0]):
        msg += "{"
        for j in range(a.shape[1]):
            msg += "%d, " % a[i,j]
            # msg += "%.5f, " % a[i,j]
        msg = msg[:-2] + "},\n"
    msg = msg[:-2] + "}"
    return msg
