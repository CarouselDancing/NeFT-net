#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint
from utils import log
import sys


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--dataset', type=str,
                                  default='h3.6m',
                                  help='path to dataset')
        self.parser.add_argument('--filename', type=str, help='path to single file to visualize')
        self.parser.add_argument('--filename_mpjpe', type=str, help='path to single file to visualize')
        self.parser.add_argument('--start_frame', type=int, help='path to single file to visualize')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true',
                                 help='whether it is to evaluate the model')
        self.parser.add_argument('--is_mpjpe', dest='is_mpjpe', action='store_true',
                                 help='whether it is to evaluate a single file through the model')

        self.parser.add_argument('--model_fold', dest='model_fold', action='store_true',
                                 help='whether it is to evaluate a single file through the model')
        self.parser.add_argument('--is_visualize', dest='is_visualize', action='store_true',
                                 help='visualize the output of the model')
        self.parser.add_argument('--is_visualize_fold', dest='is_visualize_fold', action='store_true',
                                 help='visualize the output of the folded model')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--skip_rate', type=int, default=5, help='skip rate of samples')
        self.parser.add_argument('--skip_rate_test', type=int, default=5, help='skip rate of samples for test')

        # ===============================================================
        #                     Model options
        # ===============================================================
        # self.parser.add_argument('--input_size', type=int, default=2048, help='the input size of the neural net')
        # self.parser.add_argument('--output_size', type=int, default=85, help='the output size of the neural net')

        self.parser.add_argument('--in_features', type=int, default=54, help='size of each model layer')

        #self.parser.add_argument('--in_features', type=int, default=54, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=12, help='size of each model layer')
        self.parser.add_argument('--d_model', type=int, default=256, help='past frame number')
        self.parser.add_argument('--kernel_size', type=int, default=5, help='past frame number')
        self.parser.add_argument('--drop_out', type=float, default=0.5, help='drop out probability')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--input_n', type=int, default=50, help='past frame number')
        self.parser.add_argument('--input_n_run', type=int, default=50, help='past frame number')
        self.parser.add_argument('--output_n', type=int, default=25, help='future frame number')
        self.parser.add_argument('--dct_n', type=int, default=10, help='future frame number')
        self.parser.add_argument('--lr_now', type=float, default=0.0005)
        self.parser.add_argument('--noisy', type=float, default=0)
        self.parser.add_argument('--weight_decay', type=float, default=0)
        self.parser.add_argument('--max_norm', type=float, default=10000)
        self.parser.add_argument('--epoch', type=int, default=50)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--test_batch_size', type=int, default=16)
        self.parser.add_argument('--is_load', dest='is_load', action='store_true',
                                 help='whether to load existing model')


        # ===============================================================
        #                     Augumentation
        # ===============================================================
        self.parser.add_argument('--flip_x', action= 'store_true', help='Flip X augmentation')
        self.parser.add_argument('--flip_z', action= 'store_true', help='Flip Z  augmentation')
        self.parser.add_argument('--y_rotation', type=int, default = 0, help='y rotation  augmentation')
        self.parser.add_argument('--bone_lengths', type=int,default = 0, help="Stretch parameters in the format 'stretch;<bone>;<amt>;<bone>;<amt>;...'")
        self.parser.add_argument('--flip_xz', action='store_true', help='Flip XZ  augmentation')


        #--------------------------------------------------#
        self.parser.add_argument('--shift_step', type=int, default=1, help='shift step')



    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        if not self.opt.is_eval:
            script_name = os.path.basename(sys.argv[0])[:-3]
            log_name = '{}_in{}_out{}_ks{}_dctn{}_ds{}'.format(script_name, self.opt.input_n,
                                                          self.opt.output_n,
                                                          self.opt.kernel_size,
                                                          self.opt.dct_n, self.opt.dataset)
            self.opt.exp = log_name
            # do some pre-check
            ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
                log.save_options(self.opt)
            self.opt.ckpt = ckpt
            log.save_options(self.opt)
        self._print()
        # log.save_options(self.opt)
        return self.opt
