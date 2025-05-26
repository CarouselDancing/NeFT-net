from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils
from matplotlib import pyplot as plt
import torch
import re
import csv
import math
import random
from utils import forward_kinematics

##read cvs file


#filename = "human3.6_retimed_interpolation_annotation.csv"
h36_anno_dict = {}
with open('human3.6_retimed_interpolation_annotation.csv') as h36_anno:
        reader = csv.DictReader(h36_anno)
        for row in reader:
            #annotations = row['dataset'][:-4], row['period']
            dataset = row['dataset'][:-4]
            period=  float(row['period'])
            h36_anno_dict[dataset] =period
#print(h36_anno_dict)
#exit(0)
class Datasets(Dataset):

    def __init__(self, opt, actions=None, split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        #self.path_to_data = "./datasets/h3.6m_recursive_synthetic/"
        #self.path_to_data = "./datasets/h3.6m_repeated_frames/"
        self.path_to_data = "./datasets/%s/"%(opt.dataset)
        self.split = split
        self.noise = opt.noisy
        self.offset = np.array(
                [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
                 -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
                 0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
                 0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
                 257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
                 0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
                 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
                 0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
                 0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
                 0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
        self.offset = self.offset.reshape(-1, 3)


        self.augmentation = ['none']
        if opt.flip_x:
            self.augmentation.append('flip_x')
        elif opt.flip_z:
            self.augmentation.append('flip_z')

        elif opt.y_rotation > 0:
            yrotations = ["yrot%d"%i for i in range(0, 360, int(360 / opt.y_rotation))]
            #self.augmentation.append('y_rotation')
            self.augmentation.extend(yrotations)

        elif opt.bone_lengths:
            joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RSiteF", "LeftUpLeg",
                          "LeftLeg",
                          "LeftFoot",
                          "LeftToeBase", "LSiteF", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                          "LeftForeArm",
                          "LeftHand", "LeftHandThumb", "LSiteT", "L_Wrist_End", "LSiteH", "RightShoulder", "RightArm",
                          "RightForeArm", "RightHand", "RightHandThumb", "RSiteT", "R_Wrist_End", "RSiteH"]
            # Define multiple sets of stretch factors
            self.joint_table = []

            for i, j in enumerate(joint_name):
                if j[:4] == 'Left':
                    orig = joint_name.index("".join(["Right", j[4:]]))
                elif j[:5] == 'Right':
                    orig = joint_name.index("".join(["Left", j[5:]]))
                elif j[0] == 'L':
                    orig = joint_name.index("".join(["R", j[1:]]))
                elif j[0] == 'R':
                    orig = joint_name.index("".join(["L", j[1:]]))
                else:
                    orig = i
                self.joint_table.append(orig)
            lowest_number = 0
            highest_number = len(self.joint_table) - 1
            random_number = random.randint(lowest_number, highest_number)
            #print(random_number)

            self.stretch_sets = []

            for x in range(opt.bone_lengths):
                stretch = ['stretch']

                for i in range(3):
                    bone_index = random.randint(0, len(self.joint_table) -1)
                    stretch_factor = random.randint(-25, 25)
                    stretch.append(str(bone_index))
                    stretch.append(str(stretch_factor))

                    if self.joint_table[bone_index] != bone_index: ###mirroed bone is same as the one i got i dont need to do anything

                        stretch.append(str(self.joint_table[bone_index]))
                        stretch.append(str(stretch_factor))

                self.stretch_sets.append(";".join(stretch))
                #print(self.stretch_sets)

            self.augmentation.extend(self.stretch_sets)



        #exit(0)
        if opt.flip_z and opt.flip_x:
            self.augmentation.append('flip_xz')

        self.in_n = opt.input_n_run
        self.out_n = opt.output_n
        self.sample_rate = 2
        self.p3d = {}
        self.anno = h36_anno_dict
        self.weight_decay = 1e-5

        #self.modified_template =
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        subs = np.array([np.array([1, 6, 7, 8, 9]), np.array([11]), np.array([5])], dtype= object)
        # acts = data_utils.define_actions(actions)
        if actions is None:
            # acts = ["walking", "eating", "smoking", "discussion", "directions",
            #         "greeting", "phoning", "posing", "purchases", "sitting",
            #         "sittingdown", "takingphoto", "waiting", "walkingdog",
            #         "walkingtogether"]
            acts = ["walking", "walkingtogether"]
        else:
            acts = actions
        # subs = np.array([[1], [11], [5]])
        # acts = ['walking']
        # 32 human3.6 joint name:
        # joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
        #               "LeftFoot",
        #               "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
        #               "LeftForeArm",
        #               "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
        #               "RightForeArm","RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]
        joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RSiteF", "LeftUpLeg", "LeftLeg",
                      "LeftFoot",
                      "LeftToeBase", "LSiteF", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                      "LeftForeArm",
                      "LeftHand", "LeftHandThumb", "LSiteT", "L_Wrist_End", "LSiteH", "RightShoulder", "RightArm",
                      "RightForeArm","RightHand", "RightHandThumb", "RSiteT", "R_Wrist_End", "RSiteH"]


        # [0, 6, 7, 8, 9, ... ]

        #joint_table = [i for i, j in enumerate(joint_name)]
        self.joint_table = []


        for i, j in enumerate(joint_name):
            if j[:4] == 'Left':
                orig = joint_name.index("".join(["Right", j[4:]]))
            elif j[:5] == 'Right':
                orig = joint_name.index("".join(["Left", j[5:]]))
            elif j[0] == 'L':
                orig = joint_name.index("".join(["R", j[1:]]))
            elif j[0] == 'R':
                orig = joint_name.index("".join(["L", j[1:]]))
            else:
                orig = i
            self.joint_table.append(orig)

            #print(self.joint_table)




        subs = subs[split]
        key = 0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                if self.split <= 1:
                    for subact in [1, 2]:  # subactions
                        #print(subact)
                        #exit(0)
                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)

                        modified_key = '{0}{1}_s{2}'.format(action, subact, subj)

                        period = int(h36_anno_dict[modified_key])
                        the_sequence = data_utils.readCSVasFloat(filename)
                        n, d = the_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(the_sequence[even_list, :])
                        the_sequence = torch.from_numpy(the_sequence).float().cuda()
                        # remove global rotation and translation
                        the_sequence[:, 0:6] = 0
                        #print(the_sequence.shape)

                        #p3d = torch.reshape(the_sequence[:, 3:], (-1, 32, 3))

                        ##p3d = data_utils.expmap2xyz_torch(the_sequence)
                        #print(p3d.shape)

                        # self.p3d[(subj, action, subact)] = p3d.view(num_frames, -1).cpu().data.numpy()
                        self.p3d[key] = the_sequence
                        #print(self.p3d[key].shape)
                        #exit(0)

                        valid_frames = np.arange(0, num_frames - seq_len + 1, opt.skip_rate)
                        # tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                        tmp_data_idx_1 = [key] * len(valid_frames) * len(self.augmentation)
                        #print(tmp_data_idx_1.shape)
                        #exit(0)
                        tmp_data_idx_2a = list(valid_frames)
                        tmp_data_idx_2 = []

                        qq = [[q] * len(self.augmentation) for q in tmp_data_idx_2a]
                        for z in qq:
                            tmp_data_idx_2.extend(z)

                        tmp_data_idx_3 = [period] * len(valid_frames) * len(self.augmentation)
                        tmp_data_idx_4 = self.augmentation * len(valid_frames)

                        #print("Self.Aug is %s"%self.augmentation)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, tmp_data_idx_3, tmp_data_idx_4))
                        #print(self.data_idx)
                        #exit(0)
                        key += 1
                else:
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 1)
                    #print(subact)
                    #exit(0)
                    modified_key = '{0}{1}_s{2}'.format(action, 1, subj)
                    period_1 = int(h36_anno_dict[modified_key])

                    the_sequence1 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1[even_list, :])
                    the_seq1 = torch.from_numpy(the_sequence1).float().cuda()
                    the_seq1[:, 0:6] = 0
                    #p3d1 = data_utils.expmap2xyz_torch(the_seq1)
                    #p3d1 = torch.reshape(the_seq1[:, 3:], (-1, 32, 3))

                    # self.p3d[(subj, action, 1)] = p3d1.view(num_frames1, -1).cpu().data.numpy()
                    self.p3d[key] = the_seq1

                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 2)
                    modified_key = '{0}{1}_s{2}'.format(action, 2, subj)

                    period_2 = int(h36_anno_dict[modified_key])
                    the_sequence2 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2[even_list, :])
                    the_seq2 = torch.from_numpy(the_sequence2).float().cuda()
                    the_seq2[:, 0:6] = 0
                    #p3d2 = data_utils.expmap2xyz_torch(the_seq2)
                    #p3d2 = torch.reshape(the_seq2[:, 3:], (-1, 32, 3))
                    # self.p3d[(subj, action, 2)] = p3d2.view(num_frames2, -1).cpu().data.numpy()
                    self.p3d[key + 1] = the_seq2

                    # print("action:{}".format(action))
                    # print("subact1:{}".format(num_frames1))
                    # print("subact2:{}".format(num_frames2))
                    fs_sel1, fs_sel2 = data_utils.find_indices_256(num_frames1, num_frames2, seq_len,
                                                                   input_n=self.in_n)

                    valid_frames = fs_sel1[:, 0]
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    tmp_data_idx_3 = [period_1] * len(valid_frames)
                    tmp_data_idx_4 = self.augmentation * len(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, tmp_data_idx_3, tmp_data_idx_4))

                    valid_frames = fs_sel2[:, 0]
                    tmp_data_idx_1 = [key + 1] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    tmp_data_idx_3 = [period_2] * len(valid_frames)
                    tmp_data_idx_4 = self.augmentation * len(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, tmp_data_idx_3, tmp_data_idx_4))
                    key += 2

        # ignore constant joints and joints at same position with other joints
        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        self.dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)
        #print(len(self.dimensions_to_use)) ##66
        #exit(0)

    def __len__(self):
        return np.shape(self.data_idx)[0]


    def bone_swap(self, animation):
        big_array = np.zeros_like(animation)
        for i, o in enumerate(self.joint_table):
            big_array[:, i, :] = animation[:, o, :]
        return big_array

    def __getitem__(self, item):
        if len(self.data_idx[item]) != 4:
            print("Warning, mismatched data_idx: ", self.data_idx[item])

        key, start_frame, period, augmentation = self.data_idx[item]
        print(self.data_idx[item])
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        print('Fs is ',fs)
        #exit(0)
        #print(self.p3d[key][fs].shape)
        animation = self.p3d[key][fs].cuda()

        #print(animation)
        #print('animation is ', animation.shape)
        #exit(0)

        if augmentation == 'flip_x':
            animation = self.bone_swap(animation)
            animation[:, :, 0] *= -1
            animation = data_utils.expmap2xyz_torch(animation)
        elif augmentation == 'flip_z':
            animation = self.bone_swap(animation)
            animation[:, :, 2] *= -1
            animation = data_utils.expmap2xyz_torch(animation)
        elif augmentation == 'flip_xz':
            animation[:, :, 0] *= -1
            animation[:, :, 2] *= -1
            animation = data_utils.expmap2xyz_torch(animation)
        elif augmentation.startswith('yrot'):
            degrees = int(augmentation[4:])
            theta = math.radians(degrees)
            x = animation[:, :, 0]
            z = animation[:, :, 2]
            animation[:, :, 0] = x * math.cos(theta) - z * math.sin(theta)
            animation[:, :, 2] = z * math.cos(theta) + x * math.sin(theta)
            animation = data_utils.expmap2xyz_torch(animation)
        elif augmentation.startswith('stretch'):

            multiplier = []
            if isinstance(augmentation, str) and augmentation.startswith('stretch;'):
                #print("Is this thing on?")

                stretch_params = augmentation.split(';')[1:]
                for i in range(0, len(stretch_params), 2):
                    bone_index = int(stretch_params[i])
                    multiplier_factor = (float(stretch_params[i + 1]) + 100) / 100

                    multiplier.append((bone_index, multiplier_factor))
                #print(f'Parsed bone lengths: {multiplier}')

            tpose = self.offset.copy()
            for i,m in multiplier:
                tpose[i, :] = tpose[i, :] * m



                #print('tpose multiplier',tpose.shape)
                # exit(0)

            #animation2 = data_utils.expmap2xyz_torch(animation, tpose)
            #animation2 = animation.reshape([-1, 96]).cpu().numpy()
            #print(animation.shape)
           # np.savetxt("GT_animation_variable_bone_length.txt", animation2, delimiter=',')
            animation = self.p3d[key][fs].clone()
            animation = data_utils.expmap2xyz_torch(animation)
            #animation = animation.reshape([-1, 96]).cpu().numpy()
            #print(animation2.shape)
            #np.savetxt("variable_bone_length.txt", animation, delimiter=',')
            #print("File has been saved")
            #exit(0)


        else:

            #animation = animation.reshape([animation.shape[0], -1])
            animation = data_utils.expmap2xyz_torch(animation)
            # print('animation is ',animation.shape)
            # exit(0)#should look like this torch.Size([750, 99])

        animation = animation.reshape([-1, 96]).cpu().numpy()
        #np.savetxt("variable_bone_length.txt", animation, delimiter=',')
        #print("File has been saved")
        #exit(0)
        mean = 0
        std = self.noise
        #noise = np.random.normal(mean, std, animation.shape)

        return animation, period, augmentation
