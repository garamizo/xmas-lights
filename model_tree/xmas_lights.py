######/usr/bin/env python3

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
    num_cols = 2.0
    plt.figure(figsize=(17, np.ceil(npics/num_cols) * 5))
    for i, img in enumerate(imgs):
        plt.subplot(np.ceil(npics/num_cols), num_cols, i+1)
        plt.imshow(img[:, :, [2, 1, 0]])
        colors = plt.cm.RdYlGn(np.linspace(0, 1, npts))
        plt.scatter(imgpoints[i][:, 0, 0], imgpoints[i][:, 0, 1],
                    s=150, marker='o', facecolors='none', edgecolors=colors, cmap=plt.cm.RdYlGn)
        plt.plot(imgpoints[i][:,0,0], imgpoints[i][:,0,1], color='blue')

        for j, pos in enumerate(imgpoints[i]):
            if np.isfinite(pos[0,0]):
                plt.text(pos[0,0], pos[0,1], '%d'%j, fontsize=10, color="white")



def plot_extrinsics(rvecs, tvecs, objpoints):
    """Plot 3D plots of the camera and object frame and the object points
        Returns the axes handle"""
    # avg_dist = np.mean(np.abs(objpoints.flatten()))
    avg_dist = 0.25
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

    # ax.set_aspect('box')
    # ax.set_aspect('equal', 'box')
    return ax

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

    @staticmethod
    def pack(rvecs, tvecs, objpoints):
        """Reshape extrinsic params into an optimization variable"""
        x = np.hstack((np.reshape(rvecs, -1), np.reshape(tvecs, -1), np.reshape(objpoints, -1)))
        return x

    @staticmethod
    def unpack(x, npov, npts):
        """Reshape optimization variable back into extrinsic params"""
        rvecs = list(x[:npov*3].reshape(npov, 1, 1, 3))
        tvecs = list(x[(npov*3):(npov*6)].reshape(npov, 1, 1, 3))
        objpoints = x[npov*6:].reshape(npts, 1, 3)

        # print("rvecs: %d\ntvecs: %d\nobjpoints: %d\nx: %d", 
        # print(rvecs)
        return rvecs, tvecs, objpoints

    def project_points(self, rvecs, tvecs, objpoints):
        """Project object points in the image space given the extrinsics x"""
        # fisheye: [f, c, d, tvec, rvec, skew]
        # pinhole: [rvec, tvec, f, c, d]
        # print((jac_x[0].shape))

        # cv2.fisheye.projectPoints -> imagePoints, jacobian
        #   imagePoints: (numPoints, 1, 2)
        #   jacobian: (2*numPoints, 15)

        image_points = []
        jac_x = []
        for rvec, tvec in zip(rvecs, tvecs):  # loop over each camera view
            R, _ = cv2.Rodrigues(rvec)
            # jac: f, c, k(4), rvec, tvec, alf
            #  [du1; dv1; du2; dv2; ...]
            imgp, dimgp = cv2.fisheye.projectPoints(objpoints, rvec, tvec, self.K, self.D)
            jac = dimgp[:, 8:14]
            jac_x.append(np.concatenate((jac, dimgp[:, 11:14].dot(R)), axis=1))

            # imgp, dimgp = cv2.projectPoints(objpoints, rvec, tvec, self.K, self.D)
            # jac = dimgp[:, :6]
            # jac_x.append(np.concatenate((jac, dimgp[:, :3].dot(R)), axis=1))

            image_points.append(imgp)

        # print(imgp.shape)
        return image_points, jac_x

    def cost_fcn(self, x):
        """Mean projection error squared"""
        rvecs, tvecs, objpoints = self.unpack(x, self.npov, self.npts)
        imgpoints_pred, _ = self.project_points(rvecs, tvecs, objpoints)
        # print(self.imgpoints)

        cost = 0.5 * np.nansum((np.array(self.imgpoints) - np.array(imgpoints_pred)).flatten()**2)
        return cost

    def jac_fcn(self, x):
        """Jacobian of the cost_fcn in respect to x"""
        rvecs, tvecs, objpoints = self.unpack(x, self.npov, self.npts)
        imgpoints_pred, jac_x = self.project_points(rvecs, tvecs, objpoints)

        jac_cost = 0 * x
        count = int(0)
        for i in range(self.npov):
            for j in range(self.npts):
                err = imgpoints_pred[i][j,0,:] - self.imgpoints[i][j,0,:]
                if np.any(np.isnan(err)):
                    continue

                rvec_rows = np.arange(3) + 3*i
                tvec_rows = np.arange(3) + 3*i + 3*self.npov
                s_rows = np.arange(3) + 3*j + 6*self.npov
                jac_cost[rvec_rows] += err[0] * jac_x[i][2*j,:3] + err[1] * jac_x[i][2*j+1,:3]
                jac_cost[tvec_rows] += err[0] * jac_x[i][2*j,3:6] + err[1] * jac_x[i][2*j+1,3:6]
                jac_cost[s_rows] += err[0] * jac_x[i][2*j,6:] + err[1] * jac_x[i][2*j+1,6:]
                count += 1

        # jac_cost /= count
        return jac_cost

    def extrinsic_calibration(self, useJac=True, x0=None, ub=None, lb=None):
        """Estimate x using bound-constraint optimization"""
        #TODO Add robust weighting
        
        NPOV = self.npov
        NPTS = self.npts

        # x0 = np.random.rand(NPOV*6 + NPTS*3)
        # x0 = np.array([0,0,0]*NPOV + [0,0,1]*NPOV + [0,0,0]*NPTS)
        # x0[:self.npov*3] = np.random.rand(self.npov*3) * 4*np.pi - 2*np.pi
        # if lb is None:
        #     lb = np.array([-2*np.pi]*3*NPOV + [-4, -4, 0.1]*NPOV + [-4, -4, -4]*NPTS)
        # if ub is None:
        #     ub = np.array([2*np.pi]*3*NPOV + [4, 4, 4]*NPOV + [4, 4, 4]*NPTS)
        # if x0 is None:
        #     x0 = np.random.rand(NPOV*6 + NPTS*3) * (ub - lb) + lb

        res = minimize(self.cost_fcn, x0, method='trust-constr', bounds=Bounds(lb, ub), 
                       jac=(self.jac_fcn if useJac else None),
                       options={'gtol':1e-6, 'disp': False, 'maxiter':500, 'verbose':2})
        # res = minimize(self.cost_fcn, x0, method='trust-constr', bounds=Bounds(lb, ub),
        #             options={'gtol':1e-6, 'disp': True, 'maxiter':3000, 'verbose':1})

        # bounds = [(lbi, ubi) for lbi, ubi in zip(lb, ub)]
        # res = shgo(self.cost_fcn, bounds, n=100,
        #            options={'f_tol':0.5, 'maxtime':30, 'jac':self.jac_fcn})

        rvecs = list(res.x[:self.npov*3].reshape(self.npov, 1, 1, 3))
        tvecs = list(res.x[(self.npov*3):(self.npov*6)].reshape(self.npov, 1, 1, 3))
        objpoints = res.x[self.npov*6:].reshape(1, self.npts, 3)

        return rvecs, tvecs, objpoints, res

    def plot_init_condition(self, imgs, imgpoints, x):
        """Plot images overlaied with the provided image points"""
        npts = imgpoints[0].shape[0]
        npics = len(imgs)

        rvecs, tvecs, objpoints = self.unpack(x, self.npov, self.npts)
        imgpoints_pred, jac_x = self.project_points(rvecs, tvecs, objpoints)
        # print(imgpoints_pred)
        imgpoints = imgpoints_pred

        axis = 0.1 * np.float32([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]).reshape(-1,1,3)

        num_cols = 2.0
        plt.figure(figsize=(17, np.ceil(npics/num_cols) * 5))
        for i, img in enumerate(imgs):
            plt.subplot(np.ceil(npics/num_cols), num_cols, i+1)

            axisproj, jac = cv2.fisheye.projectPoints(axis, rvecs[i], tvecs[i], self.K, self.D)
            img = draw_axis(img, axisproj)

            # plt.imshow(img[:, :, [2, 1, 0]])
            plt.imshow(img)
            colors = plt.cm.RdYlGn(np.linspace(0, 1, npts))
            plt.scatter(imgpoints[i][:, 0, 0], imgpoints[i][:, 0, 1],
                        s=150, marker='o', facecolors='none', edgecolors=colors, cmap=plt.cm.RdYlGn)
            plt.plot(imgpoints[i][:,0,0], imgpoints[i][:,0,1], color='blue')

            # pos_non = np.squeeze(imgpoints[i])
            # err = np.squeeze(imgpoints_pred[i]) - pos_non
            # plt.quiver(pos_non[:,0], pos_non[:,1], err[:,0], err[:,1], color='white', scale=1.0)

            for j, pos in enumerate(imgpoints[i]):
                if np.isfinite(pos[0,0]):
                    plt.text(pos[0,0], pos[0,1], '%d'%j, fontsize=10, color="white")
                    # posn = pos.ravel()
                    # err = imgpoints_pred[i][j].ravel() - posn
                    # plt.quiver(posn[0], )



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
            
            total period: 100 + 100 + N*200 = (N + 1)*200 ms
    """

    def __init__(self, video_path, num_bulbs=100, hsv_min=(0, 0, 100), hsv_max=(180, 255, 260)):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.nframes = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        self.flash_pattern_time = np.arange(num_bulbs) * 0.2 + 0.2

        self.THRESH = 240
        self.KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        self.SMOOTH_FILT = np.ones(5) / 5

        self.NUM_BULBS = num_bulbs
        self.HSV_MIN = hsv_min
        self.HSV_MAX = hsv_max

    def set_time(self, itime):
        iframe = round(itime * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, iframe)

    def get_time(self):
        return (self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps)

    def read(self):
        return self.cap.read()

    def find_flash_events(self, time_tol=2.0, viz=False):
        """Find brightest frame within +/- a time tolerance
            Returns time of frame occurrence
        """
        ref_time = self.get_time()
        self.set_time(ref_time - time_tol/2.0)

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
        # value = img[:, :, 2]  # goes from 0 to 255

        # _, mask = cv2.threshold(value, self.THRESH, 255, cv2.THRESH_BINARY)
        mask = (img[:,:,0] >= self.HSV_MIN[0]) & (img[:,:,0] <= self.HSV_MAX[0]) & \
                (img[:,:,1] >= self.HSV_MIN[1]) & (img[:,:,1] <= self.HSV_MAX[1]) & \
                (img[:,:,2] >= self.HSV_MIN[2]) & (img[:,:,2] <= self.HSV_MAX[2])

        # mask_rounded = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_OPEN, self.KERNEL)
        mask_rounded = mask.astype('uint8')

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
            plt.imshow(mask_rounded, cmap='gray')
            plt.show()

        elif viz_mode == 2:
            plt.figure(figsize=(12, 9))
            plt.imshow(frame[:, :, [2, 1, 0]])
            plt.plot(img_point[0], img_point[1], 'rx', markersize=area/50.0, fillstyle='none')
            plt.show()

        return img_point, area, mask_rounded


def find_tree_transf(objpoints, r00=[0,0,0], t00=[0,0,0], scale0=1.0, rad_base0=0.25, viz=False):
    """Find best transformation to fit the tree lights in a conic shape"""

    def transf_tree(objpoints, R0, t0, scale, rad_base, height=1.0):
        """Transform objpoints to new frame and calculate conic-model residuals"""
        xyz = (objpoints.reshape(-1, 3).dot(R0) * scale + t0).reshape(-1, 3)
        r = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        zh = (rad_base - r) * height / rad_base
        return xyz, zh

    R0 = cv2.Rodrigues(r00)[0]
    t0 = t00
    # scale = scale0

    def cost_fcn(x):
        # R0 = cv2.Rodrigues(x[:3])[0]
        # t0 = x[3:6]
        # scale = x[6]
        scale = x[0]
        rad_base = x[1]

        xyz, zh = transf_tree(objpoints, R0, t0, scale, rad_base)
        err = xyz[:, 2] - zh
        return np.mean(err**2) / scale**2

    objp = objpoints.reshape(-1, 3)
    maxt = np.max(np.abs(objp.flatten()))
    maxdt = np.max(np.abs(objp - np.mean(objp)).flatten())

    x0 = np.hstack((r00, t00, scale0, rad_base0))
    lb = np.array([-2*np.pi]*3 + [-maxt]*3 + [0.1/maxdt, 0.1])
    ub = np.array([2*np.pi]*3 + [maxt]*3 + [10/maxdt, 2])

    x0 = x0[6:8]
    lb = lb[6:8]
    ub = ub[6:8]

    res = minimize(cost_fcn, x0, method='trust-constr', bounds=Bounds(lb, ub), 
                    options={'gtol':1e-6, 'disp': True, 'maxiter':3000, 'verbose':1})

    # R0 = cv2.Rodrigues(res.x[:3])[0]
    # t0 = res.x[3:6]
    # scale = res.x[6]
    # rad_base = res.x[7]
    scale = res.x[0]
    rad_base = res.x[1]

    xyz, zh = transf_tree(objpoints, R0, t0, scale, rad_base)
    xyzh = np.hstack((xyz[:, :2], zh.reshape(-1, 1)))

    if viz:
        def plot_frame(ax, trans, R, name, length=1.0):
            trans = trans.reshape(-1)
            ax.quiver(trans[0], trans[1], trans[2], R[0,0], R[1,0], R[2,0], color='r', length=length)
            ax.quiver(trans[0], trans[1], trans[2], R[0,1], R[1,1], R[2,1], color='g', length=length)
            ax.quiver(trans[0], trans[1], trans[2], R[0,2], R[1,2], R[2,2], color='b', length=length)
            ax.text(trans[0], trans[1], trans[2], str("   ")+name, color='black')

        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')

        # plot_frame(ax, np.array([[0],[0],[0]]), np.eye(3), "origin")
        # ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], 'k-', lw=0.5)
        ax.plot(xyzh[:,0], xyzh[:,1], xyzh[:,2], 'r.-', lw=0.5)
        # ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2],
        #                 c=range(xyz.shape[0]), cmap=plt.cm.RdYlGn)
        # ax.quiver(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
        #           xyzh[:, 0] - xyz[:, 0], xyzh[:, 1] - xyz[:, 1], xyzh[:, 2] - xyz[:, 2])

        # ax.set(xlim=(-0.3,0.3), ylim=(-0.3,0.3), zlim=(-0.4,0.2))
        ax.view_init(0, 90)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim((-1, 1)), ax.set_ylim((-1, 1)), ax.set_zlim((-0.5, 1.5))
        plt.show()

    return R0, t0, scale, rad_base, res
    