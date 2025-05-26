import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
#from config import config
#from model import siMLPe as Model
from misc import rotmat2xyz_torch, rotmat2euler_torch, expmap2rotmat_torch

from visualize import Animation, Loader

import torch
from torch.utils.data import Dataset, DataLoader

from zed_utilities import quat_to_expmap_torch, ForwardKinematics, body_34_parts, body_34_tree, body_34_tpose, Position, \
    Quaternion, expmap_to_quat

import time

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

COMPACT_BONE_COUNT = 18
REALFULL_BONE_COUNT = 34
FULL_BONE_COUNT = 18


# Loads a single specified animation file
class AnimationSet(Dataset):
    used_joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 22, 23,
                                   24])  # Bones 32 and 33 are non-zero rotations, but constant

    def __init__(self, opt, filename, zeros=False, rotations=False, quaternions=False):
        super(AnimationSet, self).__init__()

        self.filename = filename
        self.rotations = rotations
        self.quaternions = quaternions

        if (self.quaternions):
            self.component_size = 4
        else:
            self.component_size = 3

        self.zeros = zeros

        self.h36_motion_input_length = opt.input_n
        self.h36_motion_target_length = opt.output_n

        pose = self._preprocess(self.filename)

        self.h36m_seqs = []
        self.h36m_seqs.append(pose)

        num_frames = pose.shape[0]

    def __len__(self):
        return self.h36m_seqs[0].shape[0] - (self.h36_motion_input_length + self.h36_motion_target_length)

    def __getitem__(self, index):
        start_frame = index
        end_frame_inp = index + self.h36_motion_input_length
        end_frame_target = index + self.h36_motion_input_length + self.h36_motion_target_length
        input = self.h36m_seqs[0][start_frame:end_frame_inp].float()
        target = self.h36m_seqs[0][end_frame_inp:end_frame_target].float()
        return input, target

    def _preprocess(self, filename):
        fbundle = np.load(filename, allow_pickle=True)
        if (self.rotations):
            quat = torch.tensor(fbundle['quats'].astype(np.float32))
            quat = quat[:, self.used_joint_indices, :]
            rots = quat_to_expmap_torch(quat)
            rots = np.reshape(rots, [rots.shape[0], -1])

            N = rots.shape[0]

            sample_rate = 2
            sampled_index = np.arange(0, N, sample_rate)
            h36m_zed_motion_poses = rots[sampled_index]

            T = h36m_zed_motion_poses.shape[0]
            h36m_zed_motion_poses = h36m_zed_motion_poses.reshape(T, FULL_BONE_COUNT, self.component_size)
            return h36m_zed_motion_poses
        elif (self.quaternions):
            quat = torch.tensor(fbundle['quats'].astype(np.float32))
            quat = quat[:, self.used_joint_indices, :]
            print("Comp Size is %d" % self.component_size)
            N = quat.shape[0]
            sample_rate = 2
            sampled_index = np.arange(0, N, sample_rate)
            h36m_zed_motion_poses = quat[sampled_index]
            T = h36m_zed_motion_poses.shape[0]
            h36m_zed_motion_poses = h36m_zed_motion_poses.reshape(T, FULL_BONE_COUNT, self.component_size)
        else:
            xyz_info = torch.tensor(fbundle['keypoints'].astype(np.float32))
            xyz_info = xyz_info[:, self.used_joint_indices, :]

            xyz_info = xyz_info.reshape([xyz_info.shape[0], -1])

            N = xyz_info.shape[0]

            sample_rate = 2
            sampled_index = np.arange(0, N, sample_rate)
            h36m_zed_motion_poses = 0.001 * xyz_info[sampled_index]

            T = h36m_zed_motion_poses.shape[0]
            h36m_zed_motion_poses = h36m_zed_motion_poses.reshape(T, FULL_BONE_COUNT, self.component_size)
        return h36m_zed_motion_poses

    def upplot(self, t):

        newvals = np.zeros([t.shape[0], REALFULL_BONE_COUNT, self.component_size])

        if (self.quaternions):
            newvals[:, :, 3] = 1.0

        for i, b in enumerate(self.used_joint_indices):
            newvals[:, b, :] = t[:, i, :]
        return newvals

    def fk(self, anim, quats=False):
        fk = ForwardKinematics(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)
        if (quats):
            uquats = anim
        else:
            uquats = expmap_to_quat(anim)

        big_array = np.zeros([anim.shape[0], anim.shape[1], 3])

        for i in range(uquats.shape[0]):
            rots = [Quaternion(u) for u in uquats[i]]
            xyz = fk.propagate(rots, Position([0, 0, 0]))
            big_array[i, :] = np.array([k.np() for k in xyz])

        return big_array


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


#dct_m, idct_m = get_dct_matrix(config.motion.h36m_zed_input_length_dct)
#dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
#idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)


def get_trace(config, model, dataset, frame):
    joint_used_xyz = np.arange(0, COMPACT_BONE_COUNT)
    motion_input, motion_target = dataset[frame]
    motion_input = motion_input.cuda()
    motion_input = motion_input.reshape([1, motion_input.shape[0], motion_input.shape[1], -1])

    b, n, c, _ = motion_input.shape

    motion_input = motion_input.reshape(b, n, FULL_BONE_COUNT, -1)
    motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)

    with torch.no_grad():
        trace = torch.jit.trace(model, motion_input)
    return trace


def fetch_attn(opt, model, dataset, frame):
    joint_used_xyz = np.arange(0, COMPACT_BONE_COUNT)
    motion_input, motion_target = dataset[frame]
    print(dataset[frame])
    exit(0)


    # orig_input = motion_input.clone()
    #
    # motion_input = motion_input.cuda()
    # motion_target = motion_target.cuda()
    #
    # motion_input = motion_input.reshape([1, motion_input.shape[0], motion_input.shape[1], -1])
    # motion_target = motion_target.reshape([1, motion_target.shape[0], motion_target.shape[1], -1])
    #
    # b, n, c, _ = motion_input.shape
    #
    # motion_input = motion_input.reshape(b, n, FULL_BONE_COUNT, -1)
    # motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
    #
    # outputs = []
    #
    # step = config.motion.h36m_zed_target_length_train
    #
    # if step == 25:
    #     num_step = 1
    # else:
    #     num_step = 25 // step + 1
    #
    # for idx in range(num_step):
    #     with torch.no_grad():
    #         if config.deriv_input:
    #             motion_input_ = motion_input.clone()
    #             motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_zed_input_length], motion_input_.cuda())
    #         else:
    #             motion_input_ = motion_input.clone()
    #
    #         start_time = time.time() * 1000
    #         print("Size in: ", motion_input_.shape)
    #
    #
    #         output = model(motion_input_)
    #         end_time = time.time() * 1000
    #
    #         print("Time to eval is %f" % (end_time - start_time))
    #
    #         # Should this only apply if config.deriv_input ?
    #
    #         output = torch.matmul(idct_m[:, :config.motion.h36m_zed_input_length, :], output)[:, :step, :]
    #         print("Size out: ", output.shape)
    #         if config.deriv_output:
    #             output = output + motion_input[:, -1, :].repeat(1, step, 1)
    #
    #     output = output.reshape(-1, COMPACT_BONE_COUNT * config.data_component_size)
    #     output = output.reshape(b, step, -1)
    #     outputs.append(output)
    #
    #     motion_input = torch.cat([motion_input[:, step:], output], axis=1)
    #
    # motion_pred = torch.cat(outputs, axis=1)[:, :25]
    #
    # b, n, c, _ = motion_target.shape
    #
    # motion_gt = motion_target.clone()
    # pred_rot = motion_pred.clone().reshape(b, n, COMPACT_BONE_COUNT, config.data_component_size)
    #
    # motion_pred = motion_pred.detach().cpu()
    # motion_pred = motion_target.clone().reshape(b, n, FULL_BONE_COUNT, config.data_component_size)
    # motion_pred[:, :, joint_used_xyz] = pred_rot
    #
    # tmp = motion_gt.clone()
    # tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
    #
    # motion_pred = tmp
    # # motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]
    #
    # gt_out = np.concatenate([orig_input, motion_gt.cpu().squeeze(0)], axis=0)
    # pred_out = np.concatenate([orig_input, motion_pred.cpu().squeeze(0)], axis=0)


    # gt_out and pred_out should be in [frames, 18, 3] format
    return gt_out, pred_out

def fetch(config, model, dataset, frame):
    joint_used_xyz = np.arange(0, COMPACT_BONE_COUNT)
    motion_input, motion_target = dataset[frame]

    orig_input = motion_input.clone()

    motion_input = motion_input.cuda()
    motion_target = motion_target.cuda()

    motion_input = motion_input.reshape([1, motion_input.shape[0], motion_input.shape[1], -1])
    motion_target = motion_target.reshape([1, motion_target.shape[0], motion_target.shape[1], -1])

    b, n, c, _ = motion_input.shape

    motion_input = motion_input.reshape(b, n, FULL_BONE_COUNT, -1)
    motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)

    outputs = []

    step = config.motion.h36m_zed_target_length_train

    if step == 25:
        num_step = 1
    else:
        num_step = 25 // step + 1

    for idx in range(num_step):
        with torch.no_grad():
            if config.deriv_input:
                motion_input_ = motion_input.clone()
                motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_zed_input_length], motion_input_.cuda())
            else:
                motion_input_ = motion_input.clone()

            start_time = time.time() * 1000
            print("Size in: ", motion_input_.shape)
            output = model(motion_input_)
            end_time = time.time() * 1000

            print("Time to eval is %f" % (end_time - start_time))

            # Should this only apply if config.deriv_input ?

            output = torch.matmul(idct_m[:, :config.motion.h36m_zed_input_length, :], output)[:, :step, :]
            print("Size out: ", output.shape)
            if config.deriv_output:
                output = output + motion_input[:, -1, :].repeat(1, step, 1)

        output = output.reshape(-1, COMPACT_BONE_COUNT * config.data_component_size)
        output = output.reshape(b, step, -1)
        outputs.append(output)

        motion_input = torch.cat([motion_input[:, step:], output], axis=1)

    motion_pred = torch.cat(outputs, axis=1)[:, :25]

    b, n, c, _ = motion_target.shape

    motion_gt = motion_target.clone()
    pred_rot = motion_pred.clone().reshape(b, n, COMPACT_BONE_COUNT, config.data_component_size)

    motion_pred = motion_pred.detach().cpu()
    motion_pred = motion_target.clone().reshape(b, n, FULL_BONE_COUNT, config.data_component_size)
    motion_pred[:, :, joint_used_xyz] = pred_rot

    tmp = motion_gt.clone()
    tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]

    motion_pred = tmp
    # motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]

    gt_out = np.concatenate([orig_input, motion_gt.cpu().squeeze(0)], axis=0)
    pred_out = np.concatenate([orig_input, motion_pred.cpu().squeeze(0)], axis=0)

    return gt_out, pred_out


def initialize(input_file, start_frame, quats=False, rots=False, zeros=False,
               layer_norm_axis=False,
               with_normalization=False,
               dumptrace=None
               ):
    # config.motion_mlp.with_normailzation = with_normalization
    # config.motion.norm_axis = layer_norm_axis
    #
    # if (quats):
    #     config.use_quaternions = True
    #     # config.loss_quaternion_distance = True
    #     config.motion.dim = 72  # 4 * 18
    #     config.motion_mlp.hidden_dim = config.motion.dim
    #     config.motion_fc_in.in_features = config.motion.dim
    #     config.motion_fc_in.in_features = config.motion.dim
    #     config.motion_fc_in.out_features = config.motion.dim
    #     config.motion_fc_out.in_features = config.motion.dim
    #     config.motion_fc_out.out_features = config.motion.dim
    #     config.data_component_size = 4

    #model = Model(config)



    #state_dict = torch.load(modelpth)
    # model.load_state_dict(state_dict, strict=True)
    # model.eval()
    # model.cuda()

    # config.motion.h36m_zed_target_length = config.motion.h36m_zed_target_length_eval

    # dataset = AnimationSet(config, input_file, zeros=zeros, rotations=rots, quaternions=quats)
    shuffle = False
    sampler = None
    train_sampler = None
    #
    # if (dumptrace):
    #     traced_module = get_trace(config, model, dataset, start_frame)
    #     traced_module.save(dumptrace)
    #

    gt_, pred_ = fetch_attn(dataset, start_frame)
    #
    # # print("gt_ shape: ", gt_.shape)
    # # print("pred_ shape: ", pred_.shape)
    #
    # if (rots):
    #     gt = dataset.fk(dataset.upplot(gt_))
    #     pred = dataset.fk(dataset.upplot(pred_))
    #
    # elif (quats):
    #     np.savez("predsave.npz", pred=pred_)
    #
    #     gt = dataset.fk(dataset.upplot(gt_), quats=True)
    #     pred = dataset.fk(dataset.upplot(pred_), quats=True)
    #
    # else:
    #     gt = dataset.upplot(gt_)
    #     pred = dataset.upplot(pred_)
    return gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_pth', type=str, default="./log/snapshot/model-iter-40000.pth", help='=encoder path')

    parser.add_argument("--lineplot", action='store_true', help="Draw a skel")
    parser.add_argument("--nodots", action='store_true', help="Line only, no dots")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--elev", type=float, help="Elevation", default=0)
    parser.add_argument("--azim", type=float, help="Azimuth", default=0)
    parser.add_argument("--roll", type=float, help="Roll", default=0)

    parser.add_argument("--zeros", action='store_true', help="Zero-expmap")

    parser.add_argument("--fps", type=float, default=50, help="Override animation fps")

    parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
    parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')

    parser.add_argument("--save", type=str, help="File to save animation to")
    parser.add_argument("--save_npz", type=str, help="File to save input and output files to")

    parser.add_argument('--rotations', action='store_true', help='Rotation-based data')
    parser.add_argument('--quaternions', action='store_true', help='Rotation-based data')
    parser.add_argument('--dumptrace', type=str, help="Dump the model to a Torchscript trace function for use in C++")
    parser.add_argument('file', type=str)
    parser.add_argument('start_frame', type=int)
    args = parser.parse_args()

    gt, pred = initialize(args.model_pth, args.file, args.start_frame, rots=args.rotations,
                          layer_norm_axis=args.layer_norm_axis, with_normalization=args.with_normalization,
                          dumptrace=args.dumptrace)

    if (args.save_npz):
        np.savez(args.save_npz, gt=gt, pred=pred, input_file=np.array(args.file), startframe=np.array(args.start_frame))

    anim = Animation([gt, pred], dots=not args.nodots, skellines=args.lineplot, scale=args.scale, unused_bones=True,
                     skeltype='zed', elev=args.elev, azim=args.azim, roll=args.roll, fps=args.fps, save=args.save)