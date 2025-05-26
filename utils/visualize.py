import numpy as np
import torch

import argparse
import csv

#from utils.utils import Quaternion, Vector, traverse_tree

if (__name__ == '__main__'):
    from misc import expmap2rotmat_torch,  rotmat2xyz_torch
else:    
    from misc import expmap2rotmat_torch,  rotmat2xyz_torch


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]

parents_noextra = []

body_34_names = ["PELVIS",
                 "NAVAL_SPINE",
                 "CHEST_SPINE",
                 "NECK",
                 "LEFT_CLAVICLE",
                 "LEFT_SHOULDER",
                 "LEFT_ELBOW",
                 "LEFT_WRIST",
                 "LEFT_HAND",
                 "LEFT_HANDTIP",
                 "LEFT_THUMB",
                 "RIGHT_CLAVICLE",
                 "RIGHT_SHOULDER",
                 "RIGHT_ELBOW",
                 "RIGHT_WRIST",
                 "RIGHT_HAND",
                 "RIGHT_HANDTIP",
                 "RIGHT_THUMB",
                 "LEFT_HIP",
                 "LEFT_KNEE",
                 "LEFT_ANKLE",
                 "LEFT_FOOT",
                 "RIGHT_HIP",
                 "RIGHT_KNEE",
                 "RIGHT_ANKLE",
                 "RIGHT_FOOT",
                 "HEAD",
                 "NOSE",
                 "LEFT_EYE",
                 "LEFT_EAR",
                 "RIGHT_EYE",
                 "RIGHT_EAR",
                 "LEFT_HEEL",
                 "RIGHT_HEEL"]



body_34_tree = [
    ["PELVIS", "NAVAL_SPINE"],
    ["NAVAL_SPINE", "CHEST_SPINE"],
    ["CHEST_SPINE", "LEFT_CLAVICLE"],
    ["LEFT_CLAVICLE", "LEFT_SHOULDER"],
    ["LEFT_SHOULDER", "LEFT_ELBOW"],
    ["LEFT_ELBOW", "LEFT_WRIST"],
    ["LEFT_WRIST", "LEFT_HAND"],
    ["LEFT_HAND", "LEFT_HANDTIP"],
    ["LEFT_WRIST", "LEFT_THUMB"],
    ["CHEST_SPINE", "RIGHT_CLAVICLE"],
    ["RIGHT_CLAVICLE", "RIGHT_SHOULDER"],
    ["RIGHT_SHOULDER", "RIGHT_ELBOW"],
    ["RIGHT_ELBOW", "RIGHT_WRIST"],
    ["RIGHT_WRIST", "RIGHT_HAND"],
    ["RIGHT_HAND", "RIGHT_HANDTIP"],
    ["RIGHT_WRIST", "RIGHT_THUMB"],
    ["PELVIS", "LEFT_HIP"],
    ["LEFT_HIP", "LEFT_KNEE"],
    ["LEFT_KNEE", "LEFT_ANKLE"],
    ["LEFT_ANKLE", "LEFT_FOOT"],
    ["PELVIS", "RIGHT_HIP"],
    ["RIGHT_HIP", "RIGHT_KNEE"],
    ["RIGHT_KNEE", "RIGHT_ANKLE"],
    ["RIGHT_ANKLE", "RIGHT_FOOT"],
    ["CHEST_SPINE", "NECK"],
    ["NECK", "HEAD"],
    ["HEAD", "NOSE"],
    ["NOSE", "LEFT_EYE"],
    ["LEFT_EYE", "LEFT_EAR"],
    ["NOSE", "RIGHT_EYE"],
    ["RIGHT_EYE", "RIGHT_EAR"],
    ["LEFT_ANKLE", "LEFT_HEEL"],
    ["RIGHT_ANKLE", "RIGHT_HEEL"],
    ["LEFT_HEEL", "LEFT_FOOT"],
    ["RIGHT_HEEL", "RIGHT_FOOT"]]

zed_parents = [-1] * 34

for p, c in body_34_tree:
    zed_parents[body_34_names.index(c)] = body_34_names.index(p)

class AnimationData:

    def build_frame(self, keypoints):
        numpoints = len(keypoints[0])
        t = np.array([np.ones(numpoints) * i for i in range(len(keypoints))]).flatten()

        x = keypoints[:, :, 0].reshape([-1])
        y = keypoints[:, :, 1].reshape([-1])
        z = keypoints[:, :, 2].reshape([-1])

        df = pd.DataFrame({'time' : t,
                           'x' : x,
                           'y' : y,
                           'z' : z})
        
        return df

    def unpack_extras(self, data, used):
        # Clones are bones that always seem to have the same values as other bones
        clones = {
            31 : 30,
            28 : 27,
            24 : 13,
            16 : 13,
            23 : 22,
            20 : 19
        }

        # Fixed are bones that always seem to have the same value
        fixed = { 1 : np.array([-132.9486, 0, 0]),
                  6 : np.array([132.94882, 0, 0]),
                  11 : np.array([0, 0.1, 0])}
                  
        
        retval = np.zeros([data.shape[0], 32, 3])        
        for fromi, toi in enumerate(used):
            retval[:, toi, :] = data[:, fromi, :]

        for f in fixed:
            retval[:, f, :] = fixed[f]

        for c in clones:
            retval[:, c, :] = retval[:, clones[c], :]
            
        #np.savez("unpacked_data.npz", orig = data, unpacked = retval)
        return retval


    def build_lines(self, num):
        linex = []
        liney = []
        linez = []

        
        for f in self.used_bones:
            t = self.parents[f]
            if (t >= 0):
                linex.append([self.df.x[num * self.bs + f], self.df.x[num * self.bs + t]])
                liney.append([self.df.y[num * self.bs + f], self.df.y[num * self.bs + t]])
                linez.append([self.df.z[num * self.bs + f], self.df.z[num * self.bs + t]])

        return [linex, liney, linez]
    
    def __init__(self, data, extra_bones, zedskel = False):
        
        self.zedskel = zedskel

        if (self.zedskel):
            self.bs = 34
            self.used_bones = [i for i in range(34)]
            self.extra_bones = []
            self.parents = zed_parents
            self.data = data
        else:
            self.bs = 32            
            self.used_bones = [2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]

            self.extra_bones = extra_bones

            if (not extra_bones):
                
                self.data = self.unpack_extras(data, self.used_bones)
            else:
                self.data = data

            self.parents = parents
                
        self.df = self.build_frame(self.data)

class Animation:

    def drawlines(self, aidx, frame):
        linex, liney, linez = self.animdata[aidx].build_lines(frame)
        for idx in range(len(linex)):
            self.animlines[aidx].append(self.ax[aidx].plot(linex[idx], liney[idx], linez[idx]))

    def update_plot(self, frame):

        self.framecounter.set_text("frame=%d"%frame)

        for aidx, adata in enumerate(self.animdata):
            if (self.skellines):
                linex, liney, linez = adata.build_lines(frame)
                for idx in range(len(linex)):
                    self.animlines[aidx][idx][0].set_data_3d(linex[idx], liney[idx], linez[idx])

            if (self.dots):
                newdata = adata.df[adata.df['time'] == frame]
                self.animdots[aidx]._offsets3d = (newdata.x, newdata.y, newdata.z)

            
    def __init__(self, animations, dots = True,
                 skellines = False, scale = 1.0,
                 unused_bones = True, fps = 50, save = None,
                 elev = 90.0, azim = 27.0, roll = 0.0,
                 skeltype = 'h36m',
                 grid = True
                 ):

        self.fig = plt.figure()
        self.skellines = skellines
        self.dots = dots
        self.scale = scale
        self.fps = fps
        self.ax = []

        self.gridon = grid
        
        self.elev = elev
        self.azim = azim
        self.roll = roll
        
        self.save = save

        self.extra_bones = unused_bones

        self.frames = animations[0].shape[0]

        self.skeltype = skeltype

        zs = (self.skeltype == 'zed')
        
        self.animdata = [AnimationData(anim, self.extra_bones, zedskel = zs) for anim in animations]

        self.animlines = []
        self.animdots = []

        for idx, adata in enumerate(self.animdata):
            self.ax.append(self.fig.add_subplot( 10 * len(animations) + 100 + (idx + 1), projection = '3d'))
            self.animlines.append([])
            idata = adata.df[adata.df['time'] == 0]

            if (self.skellines):
                self.drawlines(idx, 0)

            if (self.dots):
                self.animdots.append(self.ax[idx].scatter(idata.x, idata.y, idata.z))

            self.ax[idx].set_xlim(-self.scale, self.scale)
            self.ax[idx].set_ylim(-self.scale, self.scale)
            self.ax[idx].set_zlim(-self.scale, self.scale)

            self.ax[idx].view_init(elev = self.elev, azim = self.azim, roll = self.roll, vertical_axis = 'y')

            if (not self.gridon):
                self.ax[idx].grid(False)
                self.ax[idx].set_xticks([])
                self.ax[idx].set_yticks([])
                self.ax[idx].set_zticks([])
                self.ax[idx].xaxis.line.set_color((1, 1, 1, 0))
                self.ax[idx].yaxis.line.set_color((1, 1, 1, 0))
                self.ax[idx].zaxis.line.set_color((1, 1, 1, 0))                
              
        self.framecounter = plt.figtext(0.1, 0.1, "frame=0")
        print("Animation @ %d fps"%self.fps)
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames = self.frames, interval = 1000.0 / self.fps)
        if (self.save is not None):
            if (self.save[-4:] == '.gif' or self.save[-5:] == '.webp'):
                self.ani.save(filename = self.save, writer = "pillow", fps = self.fps)
            elif (self.save[-4:] == '.mkv' or self.save[-4:] == '.mp4' or self.save[-4:] == '.avi'):
                self.ani.save(filename = self.save, writer = "ffmpeg", fps = self.fps)                
            elif (self.save[-4:] == '.jpg' or self.save[-4:] == '.jpeg' or self.save[-4:] == '.png'):
                self.ani.save(filename = self.save, writer = "imagemagick", fps = self.fps)                
        plt.show()


class Loader:
    def __init__(self, filename):
        with open(args.file, "r") as fp:
            reader = csv.reader(fp)
            self.rawvals = np.array([[float(c) for c in row] for row in reader])[:, 3:]
            self.nvals = self.rawvals.reshape([self.rawvals.shape[0], 32, 3])

    def xyz(self):
        rm = expmap2rotmat_torch(torch.tensor(self.nvals.reshape(-1, 3))).float().reshape(self.nvals.shape[0], 32, 3, 3)
        return rotmat2xyz_torch(rm)

class NPZLoader:
    def __init__(self, filename):
        self.filetype = 'zed_convert'
        bundle = np.load(filename, allow_pickle = True)
        self.keypoints = bundle['keypoints']
        self.quats = bundle['quats']
        
    def xyz(self):
        return torch.tensor(self.keypoints)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type = int, help = "Scaling factor", default = 1000.0)
    parser.add_argument("--lineplot", action = 'store_true', help = "Draw a skel")
    parser.add_argument("--fps", type = float, help = "Playback fps", default = 50)
    parser.add_argument("--nodots", action = 'store_true', help = "Line only, no dots")
    parser.add_argument("--save", type = str, help = "Save to file")
    parser.add_argument("--elev", type = float, help = "Elevation", default = 0)
    parser.add_argument("--azim", type = float, help = "Azimuth", default = 0)
    parser.add_argument("--roll", type = float, help = "Roll", default = 0)
    parser.add_argument("--nogrid", action = 'store_true', help = "Remove grid")
    
    parser.add_argument("file", type = str)
    
    args = parser.parse_args()

    if (args.file[-4:] == '.csv' or args.file[-4:] == '.txt'):
        l = Loader(args.file)
        anim = Animation([l.xyz()], dots = not args.nodots, skellines = args.lineplot, scale = args.scale, fps = args.fps, save = args.save, elev = args.elev, azim = args.azim, roll = args.roll, grid = not args.nogrid)
    else:
        l = NPZLoader(args.file)
        anim = Animation([l.xyz()], dots = not args.nodots, skellines = args.lineplot, scale = args.scale, fps = args.fps, save = args.save, elev = args.elev, azim = args.azim, roll = args.roll, skeltype = 'zed', grid = not args.nogrid)
        
