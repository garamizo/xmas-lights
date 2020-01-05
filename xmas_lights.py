#!/usr/bin/env python

"""Localize bulbs from a Christmas tree."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import minimize, Bounds, shgo
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import filtfilt

def load_cam_calib(path='akaso_calib.pickle'):
    """Load camera calibration file"""

    file = open(path, 'rb')
    calib = pickle.load(file)
    file.close()

    return calib


def load_checkerboard_dataset(calib_file='akaso_calib.pickle', npov=5, npts=50):
    """Extracts image points from visual data
        Get points from checkerboard used for camera calibration
    """
    file = open(calib_file, 'rb')
    calib = pickle.load(file)
    file.close()

    num_samples = len(calib["images_path"])
    total_npts = calib["imgpoints"][0].shape[0]
    pov_selec = np.random.choice(num_samples, npov, replace=(npov > num_samples))
    pts_selec = np.random.choice(total_npts, npts, replace=(npts > total_npts))

    imgpoints = [calib["imgpoints"][s][pts_selec, :, :] for s in pov_selec]
    imgs = [cv2.imread(calib["images_path"][s]) for s in pov_selec]
    
    ground_truth = {'rvecs': [calib["rvecs"][s] for s in pov_selec],
                    'tvecs': [calib["tvecs"][s] for s in pov_selec],
                    'objpoints': calib["objpoints"][0][:, pts_selec, :]}

    return imgpoints, imgs, ground_truth


def load_xmas_tree_dataset(dataset_file='tree_dataset.pickle'):
    """Dataset contains image points and images of 50 light bulbs from 5 points of view
        It does not contains ground truth label"""
    file = open(dataset_file, 'rb')
    dataset = pickle.load(file)
    file.close()

    imgpoints = dataset["imgpoints"]
    imgs = dataset["imgs"]
    return imgpoints, imgs, None

def plot_images(imgs, imgpoints):
    """Plot images overlaied with the provided image points"""
    npts = imgpoints[0].shape[0]
    npics = len(imgs)
    plt.figure(figsize=(17, np.ceil(npics/3) * 4))
    for i, img in enumerate(imgs):
        plt.subplot(np.ceil(npics/3), 3, i+1)
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.scatter(imgpoints[i][:, 0, 0], imgpoints[i][:, 0, 1],
                    c=range(npts), cmap=plt.cm.RdYlGn)


def plot_extrinsics(rvecs, tvecs, objpoints):
    """Plot 3D plots of the camera and object frame and the object points
        Returns the axes handle"""
    avg_dist = np.mean(np.abs(objpoints.flatten()))
    def plot_frame(ax, trans, R, name, length=avg_dist):
        trans = trans.reshape(-1)
        ax.quiver(trans[0], trans[1], trans[2], R[0,0], R[1,0], R[2,0], color='r', length=length)
        ax.quiver(trans[0], trans[1], trans[2], R[0,1], R[1,1], R[2,1], color='g', length=length)
        ax.quiver(trans[0], trans[1], trans[2], R[0,2], R[1,2], R[2,2], color='b', length=length)
        ax.text(trans[0], trans[1], trans[2], str("   ")+name, color='black')

    corners = objpoints.reshape(-1, 3)

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')

    plot_frame(ax, np.array([[0],[0],[0]]), np.eye(3), "object")
    ax.plot(corners[:,0], corners[:,1], corners[:,2], 'k.-', lw=0.5)
    ax.scatter(corners[:,0], corners[:,1], corners[:,2],
                    c=range(corners.shape[0]), cmap=plt.cm.RdYlGn)

    # ax.set(xlim=(-0.3,0.3), ylim=(-0.3,0.3), zlim=(-0.4,0.2))
    ax.view_init(30, 45)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for i, (tvec, rvec) in enumerate(zip(tvecs, rvecs)):
        Rvec, _ = cv2.Rodrigues(rvec)
        R = Rvec.T
        tvec = -Rvec.T.dot(tvec.flatten())

        plot_frame(ax, tvec.flatten(), R, str(i))

    ax.set_aspect('equal', 'box')
    return ax


class Calibrator:
    """Calculate position of bulbs in Christmas Tree via computer vision
        Extrinsic solution:
            x: [rvec0, rvec1, ..., tvec0, tvec1, ..., pt0, pt1, pt2, ...]"""

    def __init__(self, imgpoints, K, D):
        
        self.imgpoints = imgpoints
        self.K = K
        self.D = D

        self.npov = len(imgpoints)
        self.npts = imgpoints[0].shape[0]

    def project_points(self, x):
        """Project object points in the image space given the extrinsics x"""
        rvecs = list(x[:self.npov*3].reshape(self.npov, 1, 1, 3))
        tvecs = list(x[(self.npov*3):(self.npov*6)].reshape(self.npov, 1, 1, 3))
        objpoints = x[self.npov*6:].reshape(self.npts, 1, 3)

        image_points = []
        jac_x = []
        for rvec, tvec in zip(rvecs, tvecs):
            R, dR = cv2.Rodrigues(rvec)
            # jac: f, c, k(4), rvec, tvec, alf
            #  [du1; dv1; du2; dv2; ...]
            imgp, dimgp = cv2.fisheye.projectPoints(objpoints, rvec, tvec, self.K, self.D)
            jac = dimgp[:, 8:14]
            jac_x.append(np.concatenate((jac, dimgp[:, 11:14].dot(R)), axis=1))
            image_points.append(imgp)

        return image_points, jac_x

    def cost_fcn(self, x):
        """Mean projection error squared"""
        imgpoints_pred, jac_x = self.project_points(x)
        cost = np.nanmean((np.array(self.imgpoints) - np.array(imgpoints_pred)).flatten()**2)
        return cost

    def jac_fcn(self, x):
        """Jacobian of the cost_fcn in respect to x"""
        NPOV = self.npov
        NPTS = self.npts
        imgpoints_pred, jac_x = self.project_points(x)

        jac_cost = 0 * x
        for i in range(NPOV):
            for j in range(NPTS):
                err = imgpoints_pred[i][j,0,:] - self.imgpoints[i][j,0,:]
                if np.any(np.isnan(err)):
                    continue

                rvec_rows = np.arange(3) + 3*i
                tvec_rows = np.arange(3) + 3*i + 3*NPOV
                s_rows = np.arange(3) + 3*j + 6*NPOV
                jac_cost[rvec_rows] += err[0] * jac_x[i][2*j,:3] + err[1] * jac_x[i][2*j+1,:3]
                jac_cost[tvec_rows] += err[0] * jac_x[i][2*j,3:6] + err[1] * jac_x[i][2*j+1,3:6]
                jac_cost[s_rows] += err[0] * jac_x[i][2*j,6:] + err[1] * jac_x[i][2*j+1,6:]

        jac_cost /= (NPOV * NPTS)  # TODO discount nan samples
        return jac_cost

    def extrinsic_calibration(self, useJac=True):
        """Estimate x using bound-constraint optimization"""
        NPOV = self.npov
        NPTS = self.npts

        x0 = np.array([0,0,0]*NPOV + [0,0,1]*NPOV + [0,0,0]*NPTS)
        lb = np.array([-2*np.pi]*3*NPOV + [-4, -4, 0.1]*NPOV + [-4, -4, -4]*NPTS)
        ub = np.array([2*np.pi]*3*NPOV + [4, 4, 4]*NPOV + [4, 4, 4]*NPTS)

        res = minimize(self.cost_fcn, x0, method='trust-constr', bounds=Bounds(lb, ub), 
                       jac=(self.jac_fcn if useJac else None),
                       options={'gtol':1e-6, 'disp': True, 'maxiter':3000, 'verbose':1})
        # res = minimize(self.cost_fcn, x0, method='trust-constr', bounds=Bounds(lb, ub),
        #             options={'gtol':1e-6, 'disp': True, 'maxiter':3000, 'verbose':1})

        # bounds = [(lbi, ubi) for lbi, ubi in zip(lb, ub)]
        # res = shgo(self.cost_fcn, bounds, n=100,
        #            options={'f_tol':0.5, 'maxtime':30, 'jac':self.jac_fcn})

        rvecs = list(res.x[:self.npov*3].reshape(self.npov, 1, 1, 3))
        tvecs = list(res.x[(self.npov*3):(self.npov*6)].reshape(self.npov, 1, 1, 3))
        objpoints = res.x[self.npov*6:].reshape(1, self.npts, 3)

        return rvecs, tvecs, objpoints, res


class TreeDetector:
    """Detect timing events of the blinking Christmas lights given an approximate video time range
        Video must contain a cycle of the flashing pattern
        Load Arduino/calibrate_xmas_tree sketch on tree, which create the following light pattern:
            Flashing routine:
            repeat:
                All on, 200 ms
                All off, 200 ms
                repeat:
                    i on, 100 ms
                    i off, 100 ms (except last loop - bug!)
                All off, 1000 ms
            
            total period: 100 + 200 + 49*200 + 1*100 + 1000 + 100
    """

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.nframes = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        self.flash_pattern_time = np.arange(50) * 0.2 + 0.35

        self.THRESH = 240
        self.KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self.SMOOTH_FILT = np.ones(5) / 5

    def set_time(self, itime):
        iframe = round(itime * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, iframe)

    def get_time(self):
        return (self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps)

    def read(self):
        return self.cap.read()

    def find_flash_events(self, time_tol=1.0, viz=False):
        """Find brightest frame within +/- a time tolerance
            Returns time of frame occurrence
        """
        ref_time = self.get_time()
        self.set_time(ref_time - time_tol)

        itime = -1
        time = []
        brights = []
        while self.cap.isOpened() and itime <= ref_time + time_tol:
            itime = self.get_time()
            ret, frame = self.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                brights.append(np.mean(frame[:, :, 2].flatten()))
                time.append(itime)

        time_interp = np.arange(ref_time-time_tol, ref_time+time_tol, 1.0/self.fps)
        brights_interp = np.interp(time_interp, np.array(time), np.array(brights))
        brights_smooth = filtfilt(self.SMOOTH_FILT, 1, brights_interp)

        pattern_start = np.argmax(brights_smooth)

        if viz:
            plt.semilogy(time_interp, brights_interp, '-')
            plt.semilogy(time_interp, brights_smooth, '-')
            plt.semilogy(time_interp[pattern_start], brights_smooth[pattern_start], 's', markersize=10)
            # plt.xlim((frange_fine[1] - 3, frange_fine[1] + 0.1))
            plt.xlabel('time [s]'), plt.ylabel('Image brightness')
            plt.legend(["interp", "smooth", "start event", "flash event"])

        return time_interp[pattern_start]

    def segment(self, frame=None, viz_mode=False):
        """Segment a frame to find a light
            plot_mode: {0: no plot, 1: plot mask, 2: plot overlay}"""
        if frame is None:
            ret, frame = self.read()
            assert ret, "Video reached EOF"

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value = img[:, :, 2]  # goes from 0 to 255

        _, mask = cv2.threshold(value, self.THRESH, 255, cv2.THRESH_BINARY)

        mask_rounded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.KERNEL)

        output = cv2.connectedComponentsWithStats(mask_rounded, connectivity=4, ltype=cv2.CV_32S)
        num_elements = output[0]
        stats = output[2]
        centroid = output[3]

        # take largest blob (that is not the background)
        if num_elements > 1:
            order = np.argsort(stats[:, 4])
            img_point = centroid[order[-2], :]
            area = stats[order[-2], 4]
        else:
            img_point = np.ones(2) * np.nan
            area = 0

        if viz_mode == 1:
            plt.figure(figsize=(12, 9))
            plt.imshow(mask_rounded, cmap=plt.cm.get_cmap('gray'))
            plt.show()

        elif viz_mode == 2:
            plt.figure(figsize=(12, 9))
            plt.imshow(frame[:, :, [2, 1, 0]])
            plt.plot(img_point[0], img_point[1], 'rx', markersize=area/50.0, fillstyle='none')
            plt.show()

        return img_point, area, mask_rounded
