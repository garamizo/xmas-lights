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



def process_video(self, videoFile):

    FLASH_PERIOD, FLASH_QTY = self.flashPeriod, self.flashQty

    # preload video at downsampled rate
    cap = cv2.VideoCapture(videoFile)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    CAP_PAUSE = 3  # downsample

    bright, frames, ftime, time = [], [], [], []
    count = -1
    while True:
        count += 1
        ret, frame = cap.read()
        if ret == False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        bright.append(np.mean(hsv[:,:,2]))
        time.append(count / FPS)
        
        if count % CAP_PAUSE == 0:
            frames.append(frame)
            ftime.append(time[-1])

    bright, time, ftime = np.array(bright), np.array(time), np.array(ftime)
    bright = signal.detrend(bright)

    # segmenting frames of interest
    tLastCycle = time[-1] - FLASH_PERIOD*FLASH_QTY
    DIFF_THRESH = (np.percentile(bright[time > tLastCycle], 200.0/FLASH_QTY) +
                    np.percentile(bright[time > tLastCycle], 50.0/FLASH_QTY)) / 2.0
    # find center of last off-region
    nStep = int(np.round(FLASH_PERIOD * FPS))
    offRegion = (bright < DIFF_THRESH)# & (time > tLastCycle)
    offEndInv = len(offRegion) - np.argmax(offRegion[::-1]) - int(nStep * (FLASH_QTY + 1.5)) # last occurrence

    rows = offEndInv + np.arange(FLASH_QTY + 1) * nStep
    self.frames = [frames[np.argmin(np.abs(ftime - t))] for t in time[rows]]
    self.bg = self.frames.pop(0)

    plt.plot(time, bright, '.-');
    plt.plot(ftime, bright[np.isin(time, ftime)], 's');
    # plt.plot(flashTime, [bright[np.argmin(np.abs(time - t))] for t in flashTime], '^');
    for t in time[rows]:
        plt.axvline(t, color='g', linestyle=':')
    plt.axhline(DIFF_THRESH, color='r', linestyle='--');
    plt.xlabel('time [s]'), plt.ylabel('brightness [0-255]'), plt.title('Frame Selection')
