import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

# from utils.misc import expmap2rotmat_torch, rotmat2xyz_torch

import torch
import torch.utils.data as data


# Dataset loader for h36m files in a zed-friendly format

def quat_to_expmap(rot_info):
    halfthetas = np.arccos(rot_info[:, :, 3])
    sinhalves = np.sin(halfthetas)
    http = np.where(sinhalves == 0, 0, 2 * halfthetas / sinhalves)
    https = np.stack([http, http, http], axis=2)
    rots = https * rot_info[:, :, :3]
    return rots


def expmap_to_quat(expmaps):
    rads = np.linalg.norm(expmaps, axis=2)
    rv = np.stack([rads, rads, rads], axis=2)
    qv = np.where(rv == 0, 0, (expmaps[:, :, :3] / rv))
    cosses = np.cos(rads / 2)
    sins = np.sin(rads / 2)
    sinss = np.stack([sins, sins, sins], axis=2)
    exps = np.concatenate([qv * sinss, np.expand_dims(cosses, 2)], axis=2)
    return exps


def quat_inverse(quats):
    exps = np.concatenate([-quats[:, :, :3], quats[:, :, 3:]], axis=2)
    return exps


def quat_mult(qa, qb):
    a = qa[:, :, 0:1]
    b = qa[:, :, 1:2]
    c = qa[:, :, 2:3]
    d = qa[:, :, 3:4]
    e = qb[:, :, 0:1]
    f = qb[:, :, 1:2]
    g = qb[:, :, 2:3]
    h = qb[:, :, 3:4]

    ww = -a * e - b * f - g * c + d * h
    ii = a * h + b * g - c * f + d * e
    jj = b * h + c * e - a * g + d * f
    kk = c * h + a * f - b * e + d * g

    qq = np.concatenate([ii, jj, kk, ww], axis=2)
    return qq


def quat_to_expmap_torch(rot_info):
    halfthetas = torch.acos(rot_info[:, :, 3])
    sinhalves = torch.sin(halfthetas)
    http = torch.where(sinhalves == 0, 0, 2 * halfthetas / sinhalves)
    https = torch.stack([http, http, http], axis=2)
    rots = https * rot_info[:, :, :3]
    return rots


def expmap_to_quat_torch(exps):
    if (len(exps.shape) == 2):
        exps = torch.reshape(exps, [exps.shape[0], -1, 3])
    rads = torch.norm(exps, dim=2)
    rv = torch.stack([rads, rads, rads], axis=2)
    qv = torch.where(rv == 0, 0, (exps[:, :, :3] / rv))
    cosses = torch.cos(rads / 2)
    sins = torch.sin(rads / 2)
    sinss = torch.stack([sins, sins, sins], axis=2)
    quats = torch.cat([qv * sinss, torch.unsqueeze(cosses, 2)], axis=2)
    return quats


# Return the rotation distance between two quaternion arrays
def quat_distance(qa, qb):
    qdiff = np.clip(quat_mult(quat_inverse(qa), qb), -1, 1)
    # Is it better to calculate sines and use np.arctan2?
    halfthetas = np.arccos(qdiff[:, :, 3])
    return 2 * halfthetas


# Return the rotation distance between two expmap arrays
def exp_distance(ea, eb):
    qa = expmap_to_quat(ea)
    qb = expmap_to_quat(eb)

    return quat_distance(qa, qb)


def quat_inverse_torch(quats):
    exps = torch.cat([-quats[:, :, :3], quats[:, :, 3:]], axis=2)
    return exps


def quat_mult_torch(qa, qb):
    a = qa[:, :, 0:1]
    b = qa[:, :, 1:2]
    c = qa[:, :, 2:3]
    d = qa[:, :, 3:4]
    e = qb[:, :, 0:1]
    f = qb[:, :, 1:2]
    g = qb[:, :, 2:3]
    h = qb[:, :, 3:4]

    # ww = -a * e - b * f - g * c + d * h

    ww0 = -a * e
    ww1 = -b * f

    np.savez("QuatMul.npz", qa=qa.cpu().detach().numpy(), qb=qb.cpu().detach().numpy())
    ww2 = -c * g
    ww3 = d * h

    ww = ww0 + ww1 + ww2 + ww3
    ii = a * h + b * g - c * f + d * e
    jj = b * h + c * e - a * g + d * f
    kk = c * h + a * f - b * e + d * g

    qq = torch.cat([ii, jj, kk, ww], axis=2)
    return qq


def quat_distance_torch(qa, qb):
    # qdiff = torch.clamp(quat_mult_torch(quat_inverse_torch(qa), qb), -1, 1)
    qdiff = torch.clamp(quat_mult_torch(qa, quat_inverse_torch(qb)), -1, 1)
    halfthetas = torch.acos(qdiff[:, :, 3])
    return 2 * halfthetas


def exp_distance_torch(ea, eb):
    qa = expmap_to_quat_torch(ea)
    qb = expmap_to_quat_torch(eb)

    return quat_distance_torch(qa, qb)


def rodrigues_torch(r):
    pass


def rodrigues_old(r):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size * angle_num, 3, 3].

    """
    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
    # theta = torch.norm(r, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float).to(r.device)
    m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
         -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float).unsqueeze(dim=0) \
              + torch.zeros((theta_dim, 3, 3), dtype=torch.float)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R


class H36MZedDataset(data.Dataset):
    def __init__(self, opt, split_name, data_aug=False, rotations=False, quaternions=False):
        super(H36MZedDataset, self).__init__()
        self._split_name = split_name
        self.data_aug = data_aug
        self._h36m_zed_anno_dir = 'datasets/h36m_zed'

        self.used_joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 22, 23,
                                            24])  # Bones 32 and 33 are non-zero rotations, but constant
        self.axis_ang = rotations
        self.quaternions = quaternions
        self._h36m_zed_files = self._get_h36m_zed_files()

        self.h36m_zed_motion_input_length = opt.input_n
        self.h36m_zed_motion_target_length = opt.output_n
        self.motion_dim = opt.in_features

        self.shift_step = opt.shift_step

        self._collect_all()
        self._file_length = len(self.data_idx)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._h36m_zed_files)

    def _get_h36m_zed_files(self):
        seq_names = []
        if (self._split_name == 'train'):
            seq_names += np.loadtxt(
                os.path.join(self._h36m_zed_anno_dir.replace('h36m_zed', ''),
                             'h36m_zed_train.txt'),
                dtype=str, ndmin=1).tolist()
        elif(self._split_name == 'test'):
            seq_names += np.loadtxt(
                os.path.join(self._h36m_zed_anno_dir.replace('h36m_zed', ''),
                             'h36m_zed_test.txt'),
                dtype=str, ndmin=1).tolist()
        else:
            seq_names += np.loadtxt(
                os.path.join(self._h36m_zed_anno_dir.replace('h36m_zed', ''),
                             'h36m_zed_valid.txt'),
                dtype=str, ndmin=1).tolist()


        print("Seq Names is %s"%seq_names)
        file_list = []
        for dataset in seq_names:
            subjects = glob.glob(self._h36m_zed_anno_dir + "/" + dataset + "/*")
            for subject in subjects:
                file_list.append(subject)

        h36m_zed_files = []

        if (self.quaternions):
            for path in file_list:
                fbundle = np.load(path, allow_pickle=True)
                quats = fbundle['quats'].astype(np.float32)[:, self.used_joint_indices, :]
                h36m_zed_files.append(torch.tensor(quats).reshape([quats.shape[0], -1]))

        elif (self.axis_ang):
            # TODO: Fix the NaN issues with this
            for path in file_list:
                fbundle = np.load(path, allow_pickle=True)
                quats = fbundle['quats'].astype(np.float32)[:, self.used_joint_indices, :]
                rots = quat_to_expmap(quats)
                rots = np.reshape(rots, [rots.shape[0], -1])
                h36m_zed_files.append(torch.tensor(rots))
        else:
            for path in file_list:
                fbundle = np.load(path, allow_pickle=True)
                xyz_info = torch.tensor(fbundle['keypoints'].astype(np.float32))
                xyz_info = xyz_info[:, self.used_joint_indices, :]
                xyz_info = xyz_info.reshape([xyz_info.shape[0], -1])
                h36m_zed_files.append(0.001 * xyz_info)

        return h36m_zed_files

    def _collect_all(self):
        self.h36m_zed_seqs = []
        self.data_idx = []
        idx = 0

        for h36m_zed_motion_poses in self._h36m_zed_files:
            N = len(h36m_zed_motion_poses)
            if (N < self.h36m_zed_motion_target_length + self.h36m_zed_motion_input_length):
                continue

            sample_rate = 2
            sampled_index = np.arange(0, N, sample_rate)
            h36m_zed_motion_poses = h36m_zed_motion_poses[sampled_index]

            T = h36m_zed_motion_poses.shape[0]
            h36m_zed_motion_poses.reshape(T, -1)

            self.h36m_zed_seqs.append(h36m_zed_motion_poses)

            valid_frames = np.arange(0, T - self.h36m_zed_motion_input_length - self.h36m_zed_motion_target_length + 1,
                                     self.shift_step)
            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))

            idx += 1

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame,
                                  start_frame + self.h36m_zed_motion_input_length + self.h36m_zed_motion_target_length)

        motion = self.h36m_zed_seqs[idx][frame_indexes]

        if self.data_aug:
            if torch.rand(1)[0] > 0.5:
                idx = [i for i in range(motion.size(0) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                motion = motion[idx]
        #print(motion.shape)
        #exit(0)

        # h36m_zed_motion_input = motion[:self.h36m_zed_motion_input_length] / 1000
        # h36m_zed_motion_target = motion[self.h36m_zed_motion_input_length:] / 1000

        h36m_zed_motion_input = motion[:self.h36m_zed_motion_input_length]
        h36m_zed_motion_target = motion[self.h36m_zed_motion_input_length:]
        h36m_zed_motion_input = h36m_zed_motion_input.float()
        h36m_zed_motion_target = h36m_zed_motion_target.float()

        # return h36m_zed_motion_input, h36m_zed_motion_target
        return motion
