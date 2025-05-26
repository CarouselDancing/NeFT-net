from utils import h36motion3d as datasets
#from utils import variable_bonelengths as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from utils import data_utils

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import csv
import h5py
import torch.optim as optim

class Loader:
    def __init__(self, filename):
        with open(filename, "r") as fp:
            reader = csv.reader(fp)
            self.rawvals = np.array([[float(c) for c in row] for row in reader])
            self.nvals = self.rawvals.reshape([self.rawvals.shape[0], -1, 3])


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features  # 66
    d_model = opt.d_model
    kernel_size = opt.kernel_size

    #
    if opt.model_fold:
        net_pred = AttModel.AttModel_fold(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    else:
        net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                num_stage=opt.num_stage, dct_n=opt.dct_n)

    net_pred.cuda()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval or opt.is_eval_fold:
        model_path_len = './{}/ckpt_last.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        #lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval or opt.is_eval_fold:
        # dataset = datasets.Datasets(opt, split=0)
        # actions = ["walking", "eating", "smoking", "discussion", "directions",
        #            "greeting", "phoning", "posing", "purchases", "sitting",
        #            "sittingdown", "takingphoto", "waiting", "walkingdog",
        #            "walkingtogether"]
        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        valid_dataset = datasets.Datasets(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=False)

    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=False)


    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
    if opt.is_eval_fold:
        ret_test = run_model_fold(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')

        return
        # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))


    #visualize
    if opt.is_visualize:

        single = single_run_model(net_pred, opt)
        return single

    if opt.is_visualize_fold:

        single_fold = single_run_model_fold(net_pred, opt)
        return single_fold

    if opt.is_mpjpe:
        mpjpe = mpjpe_dump(opt)
        return

    if opt.model_fold:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model_fold(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt, offset= 10)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid = run_model_fold(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo, offset= 10)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_test = run_model_fold(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo, offset= 10)
            print('testing error: {:.3f}'.format(ret_test['#1']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)
    #training
    # if not opt.is_eval or opt.model_fold:
    #     err_best = 1000
    #     for epo in range(start_epoch, opt.epoch + 1):
    #         is_best = False
    #         # if epo % opt.lr_decay == 0:
    #         lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2 ** (1 / opt.epoch))
    #         print('>>> training epoch: {:d}'.format(epo))
    #         ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
    #         print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
    #         ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
    #         print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
    #         ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo)
    #         print('testing error: {:.3f}'.format(ret_test['#1']))
    #         ret_log = np.array([epo, lr_now])
    #         head = np.array(['epoch', 'lr'])
    #         for k in ret_train.keys():
    #             ret_log = np.append(ret_log, [ret_train[k]])
    #             head = np.append(head, [k])
    #         for k in ret_valid.keys():
    #             ret_log = np.append(ret_log, [ret_valid[k]])
    #             head = np.append(head, ['valid_' + k])
    #         for k in ret_test.keys():
    #             ret_log = np.append(ret_log, [ret_test[k]])
    #             head = np.append(head, ['test_' + k])
    #         log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
    #         if ret_valid['m_p3d_h36'] < err_best:
    #             err_best = ret_valid['m_p3d_h36']
    #             is_best = True
    #         log.save_ckpt({'epoch': epo,
    #                        'lr': lr_now,
    #                        'err': ret_valid['m_p3d_h36'],
    #                        'state_dict': net_pred.state_dict(),
    #                        'optimizer': optimizer.state_dict()},
    #                       is_best=is_best, opt=opt)


def single_run_model(net_pred, opt=None):

    net_pred.eval()
    #exit(0)

    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    #finlename should be a command line optoion
    #start frame too to not take in the whole file
    loader = Loader(opt.filename)
    single_file = loader.rawvals
    batch_size = 1
    in_n = opt.input_n

    out_n = opt.output_n
    seq_in = opt.kernel_size
    start_f = opt.start_frame
    sample_rate = 1
    itera = 1
    #print(fs.shape)
    n, d = single_file.shape
    even_list = range(0, n, sample_rate)
    num_frames = len(even_list)
    the_sequence = np.array(single_file[even_list, :])
    the_sequence = torch.from_numpy(the_sequence).float().cuda()
    # remove global rotation and translation
    the_sequence[:, 0:6] = 0
    p3d_single = data_utils.expmap2xyz_torch(the_sequence)
    p3d_single_np = p3d_single.cpu().numpy()
    #print(p3d_single_np.shape) ###(1637, 32, 3)
    p3d_reshaped = np.reshape(p3d_single_np, [p3d_single_np.shape[0], -1])
    #print(p3d_reshaped.shape)
    #print(start_f)
    #print(in_n)
    #print(out_n)
    print(start_f, in_n, out_n)
    fs = np.arange(start_f, start_f + in_n + out_n)




    p3d_reshaped = p3d_reshaped[fs] #(60, 96)
    np.savetxt("singlefile_gt_s5_walking_1_retimed_fold.txt", p3d_reshaped, delimiter=',')
    print('in and out frame saved')
    #exit(0)
    # print(p3d_reshaped.shape)
    # #exit(0)
    single = np.expand_dims(p3d_reshaped, axis= 0)
    shape_single_torch = torch.from_numpy(single)
    shape_single = shape_single_torch.float().cuda()  ###1,60,96
    #print(len(shape_single))
    #exit(0)
    p3d_sup_single = shape_single.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
        [-1, seq_in + out_n, len(dim_used) // 3, 3])

    p3d_src_single = shape_single.clone()[:, :, dim_used]
    #print("single", p3d_src_single.shape)
    #p3d_src_single_squeeze = p3d_src_single.squeeze()
    #gt_pred = p3d_src_single_squeeze.cpu().detach().numpy()
    #print(p3d_src_single.shape)
    #exit(0)
    p3d_out_all = net_pred(p3d_src_single, input_n=in_n, output_n=out_n, itera=itera)
    #print(p3d_out_all.shape)
    #exit(0)
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    # print(index_to_equal.shape)
    # exit(0)

    p3d_out_model = shape_single.clone()[:, in_n:in_n + out_n]
    print(p3d_out_model.shape)

    p3d_out_model[:, :, dim_used] = p3d_out_all[:, seq_in:, 0]
    p3d_out_model[:, :, index_to_ignore] = p3d_out_model[:, :, index_to_equal]
    p3d_out_model_mpjpe = p3d_out_model.reshape([-1, out_n, 32, 3])

    shape_single = shape_single.reshape([-1, in_n + out_n, 32, 3])

    print(p3d_out_model_mpjpe.shape)  ###torch.Size([1, 10, 32, 3])
    #exit(0)
    p3d_out_model_squeezed = p3d_out_model.squeeze()

    #print("p3d_out_all single", p3d_out_all.shape)

    single_data_prediction = p3d_out_model_squeezed.cpu().detach().numpy()
    print(single_data_prediction.shape)
    #exit(0)
    np.savetxt("singlefile_pred_s5_walking_1_original.txt", single_data_prediction, delimiter=',')
    print('prediction saved')


    mpjpe_p3d_h36_single = torch.sum(torch.mean(torch.norm(shape_single[:, in_n:] - p3d_out_model_mpjpe, dim=3), dim=2), dim=0)
    m_p3d_h36_single = mpjpe_p3d_h36_single.cpu().data.numpy()
    m_p3d_h36_single_reshaped = np.reshape(m_p3d_h36_single, (1, -1))
    #np.savetxt("mpjpe_singlefile0.txt", m_p3d_h36_single_reshaped, delimiter=',')
    #print("MPJPE saved")
    #print(m_p3d_h36_single.shape)

    #
    # ret_single = {}
    # ret_single["m_p3d_h36"] = m_p3d_h36_single_reshaped / n
    #
    #print(m_p3d_h36_single_reshaped)
    #exit(0)
    return m_p3d_h36_single_reshaped
import os
import re
import math

def single_run_model_fold(net_pred, opt=None, offset=10):

    net_pred.eval()

    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

    loader = Loader(opt.filename)
    single_file = loader.rawvals
    batch_size = 1
    in_n = opt.input_n
    in_n_run = opt.input_n_run
    out_n = opt.output_n
    seq_in = opt.kernel_size
    start_f = opt.start_frame
    sample_rate = 1
    itera = 1

    n, d = single_file.shape
    even_list = range(0, n, sample_rate)
    the_sequence = np.array(single_file[even_list, :])
    the_sequence = torch.from_numpy(the_sequence).float().cuda()
    the_sequence[:, 0:6] = 0  # Remove global rotation and translation

    p3d_single = data_utils.expmap2xyz_torch(the_sequence)
    p3d_single_np = p3d_single.cpu().numpy()
    p3d_reshaped = np.reshape(p3d_single_np, [p3d_single_np.shape[0], -1])
    fs = np.arange(start_f, start_f + in_n_run + out_n)

    p3d_reshaped = p3d_reshaped[fs]  # Select the frames
    single = np.expand_dims(p3d_reshaped, axis=0)
    shape_single_torch = torch.from_numpy(single).float().cuda()

    p3d_sup_single = shape_single_torch[:, :, dim_used][:, -out_n - seq_in:].reshape(
        [-1, seq_in + out_n, len(dim_used) // 3, 3])

    p3d_src_single = shape_single_torch[:, :, dim_used]

    # Prepare the current, previous, and two-back windows
    window1 = p3d_src_single[:, -in_n:, :]  # Current window
    window2 = p3d_src_single[:, -in_n - offset:-offset, :]  # Previous window
    window3 = p3d_src_single[:, -in_n - 2 * offset:-2 * offset, :]  # Two windows back

    # Calculate the deltas between the windows
    delta12 = torch.norm(window1 - window2, dim=2).unsqueeze(2)
    delta23 = torch.norm(window2 - window3, dim=2).unsqueeze(2)

    # Apply weights to windows based on deltas
    window1_weighted = window1 * 1.0  # 100% influence
    window2_weighted = window2 * 0.95 * torch.exp(-delta12)  # 95% influence with delta adjustment
    window3_weighted = window3 * 0.925 * torch.exp(-delta23)  # 92.5% influence with delta adjustment

    # Concatenate the three windows along the feature dimension
    p3d_src_fold = torch.cat((window1_weighted, window2_weighted, window3_weighted), dim=2)

    # Check the shape of the concatenated tensor (should be [batch_size, in_n + out_n, 198])
    print(f"Shape of concatenated input: {p3d_src_fold.shape}")

    # Forward pass through the model
    p3d_out_all = net_pred(p3d_src_fold, input_n=in_n, output_n=out_n, itera=itera)

    # Handle the output similarly to run_model_fold
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    p3d_out_model = shape_single_torch[:, in_n:in_n + out_n]
    p3d_out_model[:, :, dim_used] = p3d_out_all[:, seq_in:, 0, :66]
    p3d_out_model[:, :, index_to_ignore] = p3d_out_model[:, :, index_to_equal]

    p3d_out_model = p3d_out_model.reshape([-1, out_n, 32, 3])

    # Further processing and saving the output
    p3d_out_model_reshape = p3d_out_model.reshape(-1, 96)
    single_data_prediction_fold = p3d_out_model_reshape.cpu().detach().numpy()

    np.savetxt("singlefile_pred_s5_walking_1_retimed_fold.txt", single_data_prediction_fold, delimiter=',')
    print('Prediction saved')

    return single_data_prediction_fold




def mpjpe_dump(opt= None):

    loader = Loader(opt.filename)
    single_file = loader.rawvals
    batch_size = 1
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size
    start_f = opt.start_frame
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    sample_rate = 1
    n, d = single_file.shape
    even_list = range(0, n, sample_rate)
    the_sequence = np.array(single_file[even_list, :])
    the_sequence = torch.from_numpy(the_sequence).float().cuda()
    p3d_single = data_utils.expmap2xyz_torch(the_sequence)
    p3d_single_np = p3d_single.cpu().numpy()
    print(p3d_single_np.shape[0])
    num_frames = p3d_single_np.shape[0] - in_n - out_n
    results = np.zeros((num_frames, out_n + 1))

    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n).cuda()
    for i in range(num_frames):
        opt.start_frame = i
        single_file_mpjpes = single_run_model(net_pred, opt)
        #print(single_file_mpjpes)
        results[i, 0] = i
        results[i, 1:] = single_file_mpjpes

        np.savetxt("MPJPE_retimed_interpolation_s5_walkingtogether_1.txt", results, delimiter=',')
        #np.savetxt("mpjpe_h3.6m_periodic_S5_walking_1.txt", results, delimiter=',')
        print("MPJPE results saved to mpjpe_singlefile0.txt")

def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):

    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()


    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    seq_in = opt.kernel_size
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    itera = 1
    idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
            out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    st = time.time()
    for i, (p3d_h36, ano, aug) in enumerate(data_loader):


        # if ('flip_x' in aug):
        #     print("Saving flipped data and aborting")
        #     idx = aug.index('flip_x')
        #     np.savetxt("try_fold.txt", p3d_h36[idx], delimiter=',')
        #     exit(0)
        #print(ano)
        #exit(0)
        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().cuda()
        #print("The shape of p3d_h3d is", p3d_h36.shape)
        #exit(0)
        p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_src = p3d_h36.clone()[:, :, dim_used]


        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera)
        #print("p3d_out_all model out", p3d_out_all.shape)
        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        #print(p3d_out.shape)
        p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:, 0]
        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]

        p3d_out = p3d_out.reshape([-1, out_n, 32, 3]) ####torch.Size([32, 10, 32, 3])
        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])
        #print(p3d_h36.shape)
        #exit(0)
        p3d_out_all = p3d_out_all.reshape([batch_size, seq_in + out_n, itera, len(dim_used) // 3, 3])
        #print("p3d_out_all", p3d_out_all.shape)
        #exit(0)
        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0] - p3d_sup, dim=3))
            loss_all = loss_p3d
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d.cpu().data.numpy() * batch_size


            ###fk on the model out put

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            print(p3d_out.shape)
            exit(0)
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size

        else:
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))
    ret = {}
    #print(ret)
    #exit(0)
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    return ret


def run_model_fold(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, offset=0):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n

    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    seq_in = opt.kernel_size

    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    itera = 1
    idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
            out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    st = time.time()

    for i, (p3d_h36, ano, aug) in enumerate(data_loader):

        batch_size, seq_n, _ = p3d_h36.shape
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().cuda()

        p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])

        # Extract the current, previous, and two-back windows
        p3d_src = p3d_h36.clone()[:, :, dim_used]

        # Initialize the second and third window (history slices)
        p3d_src_slice_his = torch.zeros([p3d_src.shape[0], in_n + out_n, p3d_src.shape[2]]).cuda()
        p3d_src_slice_third = torch.zeros([p3d_src.shape[0], in_n + out_n, p3d_src.shape[2]]).cuda()

        # First window (current)
        p3d_src_slice_updated = p3d_src[:, -in_n - out_n:, :]

        # Second window (previous)
        for i, p in enumerate(ano):
            p3d_src_slice_his[i, :, :] = p3d_src[i, -in_n - out_n - p - offset:-p - offset, :]

        # Third window (two steps back)
        for i, p in enumerate(ano):
            p3d_src_slice_third[i, :, :] = p3d_src[i, -in_n - out_n - p - 2 * offset:-p - 2 * offset, :]

        p3d_src_fold = torch.cat((p3d_src_slice_updated, p3d_src_slice_his, p3d_src_slice_third), dim=2)

        p3d_out_all = net_pred(p3d_src_fold, input_n=in_n, output_n=out_n, itera=itera)

        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:, 0, :66]
        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]

        p3d_out = p3d_out.reshape([-1, out_n, 32, 3])
        p3d_h36_slice = p3d_h36[:, -in_n - out_n:, :].reshape([-1, in_n + out_n, 32, 3])

        p3d_out_all = p3d_out_all[:, :, :, :66].reshape([batch_size, seq_in + out_n, itera, len(dim_used) // 3, 3])

        # 2D joint loss:
        grad_norm = 0
        if is_train == 0:
            loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0] - p3d_sup, dim=3))
            loss_all = loss_p3d
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            l_p3d += loss_p3d.cpu().data.numpy() * batch_size

        if is_train <= 1:
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36_slice[:, -out_n:] - p3d_out, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36_slice[:, -out_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))

    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
