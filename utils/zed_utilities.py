import numpy as np
import torch
import argparse
import math

import argparse

from scipy.spatial.transform import Rotation

import torch

body_34_parts = [
    "PELVIS",
    "NAVALSPINE",
    "CHESTSPINE",
    "NECK",
    "LEFTCLAVICLE",
    "LEFTSHOULDER",
    "LEFTELBOW",
    "LEFTWRIST",
    "LEFTHAND",
    "LEFTHANDTIP",
    "LEFTTHUMB",
    "RIGHTCLAVICLE",
    "RIGHTSHOULDER",
    "RIGHTELBOW",
    "RIGHTWRIST",
    "RIGHTHAND",
    "RIGHTHANDTIP",
    "RIGHTTHUMB",
    "LEFTHIP",
    "LEFTKNEE",
    "LEFTANKLE",
    "LEFTFOOT",
    "RIGHTHIP",
    "RIGHTKNEE",
    "RIGHTANKLE",
    "RIGHTFOOT",
    "HEAD",
    "NOSE",
    "LEFTEYE",
    "LEFTEAR",
    "RIGHTEYE",
    "RIGHTEAR",
    "LEFTHEEL",
    "RIGHTHEEL"
]


body_38_parts = [
    "PELVIS",
    "SPINE_1",
    "SPINE_2",
    "SPINE_3",
    "NECK",
    "NOSE",
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_EAR",
    "RIGHT_EAR",
    "LEFT_CLAVICLE",
    "RIGHT_CLAVICLE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_BIG_TOE",
    "RIGHT_BIG_TOE",
    "LEFT_SMALL_TOE",
    "RIGHT_SMALL_TOE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_HAND_THUMB_4",
    "RIGHT_HAND_THUMB_4",
    "LEFT_HAND_INDEX_1",
    "RIGHT_HAND_INDEX_1",
    "LEFT_HAND_MIDDLE_4",
    "RIGHT_HAND_MIDDLE_4",
    "LEFT_HAND_PINKY_1",
    "RIGHT_HAND_PINKY_1"]

body_34_tree = { 
    "PELVIS": ["NAVALSPINE", "LEFTHIP", "RIGHTHIP"],
    "NAVALSPINE" : ["CHESTSPINE"],
    "CHESTSPINE" : ["LEFTCLAVICLE", "RIGHTCLAVICLE", "NECK"],

    "LEFTCLAVICLE" : ["LEFTSHOULDER"],
    "LEFTSHOULDER" : ["LEFTELBOW"],
    "LEFTELBOW" : ["LEFTWRIST"],
    "LEFTWRIST" : ["LEFTHAND", "LEFTTHUMB"],
    "LEFTHAND" : ["LEFTHANDTIP"],
     
    "RIGHTCLAVICLE" : ["RIGHTSHOULDER"],
    "RIGHTSHOULDER" : ["RIGHTELBOW"],
    "RIGHTELBOW" : ["RIGHTWRIST"],
    "RIGHTWRIST" : ["RIGHTHAND", "RIGHTTHUMB"],
    "RIGHTHAND" : ["RIGHTHANDTIP"],
     
    "LEFTHIP" : ["LEFTKNEE"],
    "LEFTKNEE" : ["LEFTANKLE"],
    "LEFTANKLE" : ["LEFTFOOT", "LEFTHEEL"],
    "LEFTHEEL" : ["LEFTFOOT"],
    
    "RIGHTHIP" : ["RIGHTKNEE"],
    "RIGHTKNEE" : ["RIGHTANKLE"],
    "RIGHTANKLE" : ["RIGHTFOOT", "RIGHTHEEL"],
    "RIGHTHEEL" : ["RIGHTFOOT"],

    "NECK" : ["HEAD", "LEFTEYE", "RIGHTEYE"],
    "HEAD" : ["NOSE"],
    "LEFTEYE" : ["LEFTEAR"],
    "RIGHTEYE" : ["RIGHTEAR"],

    "LEFTHANDTIP" : [],
    "LEFTTHUMB" : [],
    
    "RIGHTHANDTIP" : [],
    "RIGHTTHUMB" : [],
    
    "NOSE" : [],
    "LEFTEAR" : [],
    "RIGHTEAR" : [],
    
    "LEFTFOOT" : [],
    "RIGHTFOOT" : []

    
    }

body_38_tree = {
    "PELVIS": ["SPINE1", "LEFTHIP", "RIGHTHIP"],
    
    "SPINE1": ["SPINE2"],
    "SPINE2": ["SPINE3"],
    "SPINE3": ["NECK", "LEFTCLAVICLE", "RIGHTCLAVICLE"],

    "NECK": ["NOSE"],
    "NOSE": ["LEFTEYE", "RIGHTEYE"],
    "LEFTEYE": ["LEFTEAR"],
    "RIGHTEYE": ["RIGHTEAR"],
    
    "LEFTCLAVICLE": ["LEFTSHOULDER"],
    "LEFTSHOULDER": ["LEFTELBOW"],
    "LEFTELBOW": ["LEFTWRIST"],
    "LEFTWRIST": ["LEFTHANDTHUMB4",
                   "LEFTHANDINDEX1",
                   "LEFTHANDMIDDLE4",
                   "LEFTHANDPINKY1"],

    "RIGHTCLAVICLE": ["RIGHTSHOULDER"],
    "RIGHTSHOULDER": ["RIGHTELBOW"],
    "RIGHTELBOW": ["RIGHTWRIST"],
    "RIGHTWRIST": ["RIGHTHANDTHUMB4",
                   "RIGHTHANDINDEX1",
                   "RIGHTHANDMIDDLE4",
                   "RIGHTHANDPINKY1"],
    
    "LEFTHIP" : ["LEFTKNEE"],
    "LEFTKNEE" : ["LEFTANKLE"],
    "LEFTANKLE" : ["LEFTHEEL", "LEFTBIGTOE", "LEFTSMALLTOE"],
    
    "RIGHTHIP" : ["RIGHTKNEE"],
    "RIGHTKNEE" : ["RIGHTANKLE"],
    "RIGHTANKLE" : ["RIGHTHEEL", "RIGHTBIGTOE", "RIGHTSMALLTOE"],

    
}


# Utility functions for handling Zed data

body_34_tpose = [[0,0,0],
                 [-0.000732270938924443,175.158289701814,0.0000404],
                 [0.104501423707752,350.306093180137,0.061023624087894],
                 [0.209734388920577,525.459141360196,0.122005361332611],
                 [-47.5920764358155,526.439401661188,0.945479045649915],
                 [-173.508163386225,526.509589382348,2.98848444604651],
                 [-413.490495136004,529.070006884027,4.30015451481757],
                 [-644.259234923473,531.557805055099,5.5156119864156],
                 [-690.412981869986,532.055366973068,5.75870846651414],
                 [-782.7204827236,533.050484095638,6.24488492674942],
                 [-737.171470366004,477.177940319642,6.03507728101326],
                 [48.0126939292815,526.382576868867,-0.700813930486266],
                 [173.928779031094,526.312389136708,-2.74381818341968],
                 [413.965281719423,525.913766951398,-5.58751914341834],
                 [644.770225962532,525.03114718918,-8.36370311424063],
                 [690.931216942206,524.854620798345,-8.91894225928869],
                 [783.253196286002,524.501573309859,-10.029413939978],
                 [736.897241876494,469.296359031723,-9.22217198040476],
                 [-97.2538992002065,0,-0.0216466824273884],
                 [-97.2534603765872,-398.665678321646,-0.0280368622539885],
                 [-97.236806268837,-753.034804644565,-0.0408179870540185],
                 [-97.2541801180481,-841.630928132314,106.266711069826],
                 [97.2538982249762,0,0.0216480069085377],
                 [97.2606469820357,-398.664829041876,0.0265545153545871],
                 [97.2769776133806,-753.033961966181,0.0252733646276821],
                 [97.2596464463626,-841.626635633415,106.335670500348],
                 [1.30731388292955,660.085767955841,62.6295143653575],
                 [1.37823874062583,704.938429698234,62.5519366785149],
                 [-25.8996591150869,736.342683178085,31.5073346749533],
                 [-76.4225396149832,715.693774717041,-52.9081045914678],
                 [27.8706194688158,736.259256448681,30.7476818991719],
                 [75.9266183557466,715.457396322932,-55.0604622597071],
                 [-97.2254670188798,-841.625806800333,-35.4809212169046],
                 [97.2881989464546,-841.626116231009,-35.4119624250308]]

def quat_to_expmap(rot_info):
    halfthetas = np.arccos(rot_info[:, :, 3])
    sinhalves = np.sin(halfthetas)
    http = np.where(sinhalves == 0, 0, 2 * halfthetas/sinhalves)
    https = np.stack([http, http, http], axis = 2)
    rots = https * rot_info[:, :, :3]
    return rots

def expmap_to_quat(expmaps):
    rads = np.linalg.norm(expmaps, axis = 2)
    rv = np.stack([rads, rads, rads], axis = 2)
    qv = np.where(rv == 0, 0, (expmaps[:, :, :3] / rv))
    cosses = np.cos (rads / 2)
    sins = np.sin(rads / 2)
    sinss = np.stack([sins, sins, sins], axis = 2)
    exps = np.concatenate([qv * sinss , np.expand_dims(cosses, 2)], axis = 2)
    return exps

def quat_inverse(quats):
    exps = np.concatenate([-quats[:, :, :3], quats[:, :, 3:]], axis = 2)    
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

    qq = np.concatenate([ii, jj, kk, ww], axis = 2)
    return qq

def quat_to_expmap_torch(rot_info):
    halfthetas = torch.acos(rot_info[:, :, 3])
    sinhalves = torch.sin(halfthetas)
    http = torch.where(sinhalves == 0, 0, 2 * halfthetas/sinhalves)
    https = torch.stack([http, http, http], axis = 2)
    rots = https * rot_info[:, :, :3]
    return rots

def expmap_to_quat_torch(exps):
    if (len(exps.shape) == 2):
        exps = torch.reshape(exps, [exps.shape[0], -1, 3])
    rads = torch.norm(exps, dim = 2)
    rv = torch.stack([rads, rads, rads], axis = 2)
    qv = torch.where(rv == 0, 0, (exps[:, :, :3] / rv))
    cosses = torch.cos (rads / 2)
    sins = torch.sin(rads / 2)
    sinss = torch.stack([sins, sins, sins], axis = 2)
    quats = torch.cat([qv * sinss , torch.unsqueeze(cosses, 2)], axis = 2)
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
    exps = torch.cat([-quats[:, :, :3], quats[:, :, 3:]], axis =2)
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

    ww = -a * e - b * f - g * c + d * h
    ii = a * h + b * g - c * f + d * e
    jj = b * h + c * e - a * g + d * f
    kk = c * h + a * f - b * e + d * g

    qq = torch.cat([ii, jj, kk, ww], axis = 2)
    return qq

def quat_distance_torch(qa, qb):
    qdiff = torch.clamp(quat_mult_torch(quat_inverse_torch(qa), qb), -1, 1)
    halfthetas = torch.acos(qdiff[:, :, 3])
    return 2 * halfthetas

def exp_distance_torch(ea, eb):
    qa = expmap_to_quat_torch(ea)
    qb = expmap_to_quat_torch(eb)

    return quat_distance_torch(qa, qb)


class Quantized_Quaternion:
    # Represent a quaternion with three 16-bit fixed-point ints
    def __init__(self, ints):
        self.fixed = ints

    def toQuaternion(self):
        floats = [f / 32767 for f in self.fixed]
        sqrs = [f * f for f in floats]
        # print("Sqrs is ", sqrs)
        # print("Sumsq is %f"%(1.0 - sum(sqrs)))
        floats.append(math.sqrt(1.0 - sum(sqrs)))
        return Quaternion(floats)

    def zero():
        return Quantized_Quaternion([0.0, 0.0, 0.0])

    def __str__(self):
        return Quaternion.toQuaternion.__str__()

    def np(self):
        return np.array(self.fixed).astype(np.int16)
    
class Quaternion:

    def __init__(self, floats):
        self.rot = Rotation.from_quat(floats)

    def __mul__(self, q):
        rmul = (self.rot * q.rot).as_quat()
        return Quaternion(rmul)

    def zero():
        return Quaternion([0.0, 0.0, 0.0, 1.0])
        
    def toEuler(self, perm = 'xyz'):
        e = self.rot.as_euler(perm, degrees = True)
        return Euler([e[0], e[1], e[2]])

    def cstr(self, sep = ","):
        q = self.rot.as_quat()
        return (sep.join([str(i) for i in q]))
    
    def __str__(self):
        q = self.rot.as_quat()
        return (" ".join([str(i) for i in q]))

    def apply(self, x):
        return Position(self.rot.apply([x.x, x.y, x.z]))

    def np(self):
        return self.rot.as_quat()

    def torch(self):
        return torch.tensor(self.rot.as_quat())
    
    def toQuantQuat(self):
        q = self.rot.as_quat()
        if (q[3] < 0):
            qq = -q
        else:
            qq = q            
        ints = [round(32767 * f) for f in qq]
        return Quantized_Quaternion(ints)
    
class Euler:
    def __init__(self, floats, perm = 'xyz'):

        self.perm = perm
        self.e0 = floats[0]
        self.e1 = floats[1]
        self.e2 = floats[2]

    def cstr(self, sep = ","):
        return (sep.join([str(i) for i in [self.e0, self.e1, self.e2]]))
    
    def __str__(self):
        return (" ".join([str(i) for i in [self.e0, self.e1, self.e2]]))

    def toQuat(self, xneg = False, zneg = False):

        rot = Rotation.from_euler(self.perm, [self.e0,
                                              self.e1,
                                              self.e2], degrees = True)
        q = rot.as_quat()

        
        if (xneg == True):
            xcoord = -q[0]
        else:
            xcoord = q[0]
        if (zneg == True):
            zcoord = -q[2]
        else:
            zcoord = q[2]

        return Quaternion([xcoord, q[1], zcoord, q[3]])


    def toQuantQuat(self, xneg = False, zneg = False):
        return self.toQuat(xneg = xneg, zneg = zneg).toQuantQuat()

class Position:
    def __init__(self, floats):
        [self.x, self.y, self.z] = floats

    def cstr(self, sep = ","):
        return (sep.join([str(i) for i in [self.x, self.y, self.z]]))
    
    def __str__(self):
        return (" ".join([str(i) for i in [self.x, self.y, self.z]]))


    def scale(self, s):
        return Position([s * self.x , s * self.y, s * self.z])

    def __add__(self, a):
        return Position([self.x + a.x, self.y + a.y, self.z + a.z])

    def __sub__(self, a):
        return Position([self.x - a.x, self.y - a.y, self.z - a.z])

    def __mul__ (self, k):
        return Position([k * self.x, k * self.y, k * self.z])

    
    def np(self):
        return np.array([self.x, self.y, self.z])

    def torch(self):
        return torch.tensor([self.x, self.y, self.z])
    
    
class Transform:
    def __init__(self, pos, ori):
        self.pos = pos
        self.ori = ori

    def cstrquat(self, sep=","):
        return "%s%s%s"%(self.pos.cstr(sep),sep,self.ori.cstr(sep))
    
    def cstr(self, sep=","):
        return "%s%s%s"%(self.pos.cstr(sep),sep,self.ori.toEuler(args.convert_order).cstr(sep))
    
    def __str__(self):
        return "%s %s"%(self.pos,self.ori.toEuler(args.convert_order))

    def scale(self, x):
        return Transform(self.pos.scale(x), self.ori)

    def offset_pos(self, p):
        return Transform(self.pos + p, self.ori)


# def expmap2rotmat_torch(r):
#     """
#     Converts expmap matrix to rotation
#     batch pytorch version ported from the corresponding method above
#     :param r: N*3
#     :return: N*3*3
#     """
#     theta = torch.norm(r, 2, 1)

#     r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
#     r1 = torch.zeros_like(r0).repeat(1, 3)
#     r1[:, 1] = -r0[:, 2]
#     r1[:, 2] = r0[:, 1]
#     r1[:, 5] = -r0[:, 0]
#     r1 = r1.view(-1, 3, 3)
#     r1 = r1 - r1.transpose(1, 2)
#     n = r1.data.shape[0]
#     R = torch.eye(3, 3).repeat(n, 1, 1).float().to(r.device) + torch.mul(
#         torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
#         (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
#     return R

# Takes a batched tensor of rotations in [batch, frame, bone, 3, 3] format
# Apply the to the vertex


def old_rotate_vector(v, k, theta):
    v = np.asarray(v)
    k = np.asarray(k) / np.linalg.norm(k)  # Normalize k to a unit vector
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    q
    term1 = v * cos_theta
    term2 = np.cross(k, v) * sin_theta
    term3 = k * np.dot(k, v) * (1 - cos_theta)


    print("Precross k: ", k)
    print("Precross v: ", v)
    print("Precross Sin: ", sin_theta)
    
    print("Cross Product is ", np.cross(k, v))
    return term1 + term2 + term3, term1, term2, term3


def batch_rotate_vector(quats, bone, vector):

    """ Rotates a vector by a batch of quaternions using Rodriguez rotation formula
    - 'quats': Double-batched quaternions in [batch, frame, bone, 4] shape
    - bone - bone index
    - 'vector' : 3-vector Vector to be rotated by the given bone quaternion
    """
    halftheta = torch.acos(quats[:, :, :, 3])
    #halftheta = torch.acos(quats[:, :, bone:bone + 1, 3])
    
    sinhalves = torch.unsqueeze(torch.sin(halftheta), dim = 2)

    kvecs = torch.div(quats[:, :, :, :3], sinhalves)
    #kvecs = torch.div(quats[:, :, bone:bone + 1, :3], sinhalves)
    
    sines = torch.unsqueeze(torch.sin(2 * halftheta), dim = 2)

    costheta = torch.unsqueeze(torch.cos(2 * halftheta), dim = 2)

    t1 = costheta * vector

    t2 = torch.cross(kvecs, vector.expand_as(kvecs), dim = 3) * sines

    dotproduct = torch.sum(kvecs * vector.expand_as(kvecs), dim = 3)
    t3 = kvecs * torch.unsqueeze(dotproduct, dim = 3) * (1 - costheta)

    outval = t1 + t2 + t3

    # if it's a Nan here, it's because it's a 0-rotation quaternion, most likely
    return torch.where(torch.isnan(outval), vector, outval)


# Takes a batched set of tensors and another one and quaternion-multiply them
def batch_quat_multiply(qa, qb, cIdx = None):
    if (cIdx is None):
        a = qa[:, :, :, 0:1]
        b = qa[:, :, :, 1:2]
        c = qa[:, :, :, 2:3]
        d = qa[:, :, :, 3:4]
        
        e = qb[:, :, :, 0:1]
        f = qb[:, :, :, 1:2]
        g = qb[:, :, :, 2:3]
        h = qb[:, :, :, 3:4]
        
        ww = -a * e - b * f - g * c + d * h
        ii = a * h + b * g - c * f + d * e
        jj = b * h + c * e - a * g + d * f
        kk = c * h + a * f - b * e + d * g
        
        qq = torch.cat([ii, jj, kk, ww], axis = 3)
        return qq

    else:
        a = qa[:, :, cIdx:cIdx + 1, 0:1]
        b = qa[:, :, cIdx:cIdx + 1, 1:2]
        c = qa[:, :, cIdx:cIdx + 1, 2:3]
        d = qa[:, :, cIdx:cIdx + 1, 3:4]
        
        e = qb[:, :, :, 0:1]
        f = qb[:, :, :, 1:2]
        g = qb[:, :, :, 2:3]
        h = qb[:, :, :, 3:4]
        
        ww = -a * e - b * f - g * c + d * h
        ii = a * h + b * g - c * f + d * e
        jj = b * h + c * e - a * g + d * f
        kk = c * h + a * f - b * e + d * g

        print("II shape is: ", ii.shape)
        print("TC shape is ",torch.cat([ii, jj, kk, ww], axis = 3).shape)
        qq = qa.clone()
        qq[:, :, cIdx:cIdx + 1, :] = torch.cat([ii, jj, kk, ww], axis = 3)
        return qq
        
       
class ForwardKinematics:

    def __init__(self, bonelist, bonetree, rootbone, tpose, rootpos = Position([0,0,0])):
        self.bonetree = bonetree
        self.bonelist = bonelist
        self.root = rootbone
        self.tpose = [Position(p) for p in tpose]
        
    def propagate(self, rotations, initial_position):
        keyvector = [Position([0, 0, 0]) for i in range(34)]
        
        def _recurse(bone, c_rot, pIdx):
            cIdx = self.bonelist.index(bone)

            if (pIdx < 0):
                n_rot = c_rot
                new_pos = initial_position
            else:
                n_rot = c_rot * rotations[pIdx]
                new_pos = keyvector[pIdx] + n_rot.apply(self.tpose[cIdx] - self.tpose[pIdx])
                # print("Old: %d, Nrot:"%cIdx, n_rot)
                # print("Old: %d, NewPos: "%cIdx, new_pos)


            keyvector[cIdx] = new_pos

            for child in self.bonetree[bone]:
                _recurse(child, n_rot, cIdx)
                
        initial_rot = rotations[self.bonelist.index(self.root)]

        _recurse(self.root, initial_rot, -1)
        
        return keyvector


class ForwardKinematics_Torch:

    def __init__(self, bonelist, bonetree, rootbone, tpose, rootpose = torch.tensor([0, 0, 0])):
        self.bonelist = bonelist
        self.bonetree = bonetree
        self.root = rootbone
        self.tpose = torch.tensor(tpose).cuda()


    def propagate(self, rotations, initial_position = None):

        if (initial_position):
            ipos = initial_position
        else:
            ipos = torch.zeros([3])

        key_tensor = torch.zeros([rotations.shape[0], rotations.shape[1], rotations.shape[2], 3]).cuda()
        def _recurse(parentbone, cur_rot, pIdx):

            cIdx = self.bonelist.index(parentbone)

            if pIdx < 0:
                new_rot = cur_rot.clone()
                new_pos = ipos.clone()
            else:
                new_rot = batch_quat_multiply(cur_rot, rotations[:, :, pIdx:pIdx + 1, :])
                brv = batch_rotate_vector(new_rot, pIdx, self.tpose[cIdx] - self.tpose[pIdx])
                new_pos = key_tensor[:, :, pIdx:pIdx + 1, :] + brv

            key_tensor[:, :, cIdx:cIdx + 1, :] = new_pos

            for child in self.bonetree[parentbone]:
                _recurse(child, new_rot, cIdx)

        iidx = self.bonelist.index(self.root)
        initial_rot = rotations[:, :, iidx:iidx + 1, :]
        _recurse(self.root, initial_rot, -1)
        return key_tensor
        

def normalize(v):
    norm = np.linalg.norm(v, axis = 1)
    nms = np.stack([norm, norm, norm]).T
    return v / nms
    
                
class PointsToRotations:
    
    def __init__(self, keypoints, body_names, body_tree, root_bone = 'PELVIS'):
        self.keypoints = keypoints
        self.tree = body_tree
        self.names = body_names
        self.root = root_bone
        self.root_idx = self.names.index(self.root)

        
    def root_rot(self):

        root_pos = self.keypoints[:, self.root_idx, :]
        lhip_idx = [self.names.index(i) for i in self.tree[self.root] if 'left' in i.lower()][0]
        rhip_idx = [self.names.index(i) for i in self.tree[self.root] if 'right' in i.lower()][0]
        spine_idx = [self.names.index(i) for i in self.tree[self.root] if 'spine' in i.lower()][0]
        
        lhval = self.keypoints[:, lhip_idx, :]
        rhval = self.keypoints[:, rhip_idx, :]        
        sval = self.keypoints[:, spine_idx, :]
        

        leftvec = normalize(lhval - root_pos)
        upvec = normalize(sval - root_pos)
        
        fwdvec = np.cross(upvec, leftvec)

        return np.stack([leftvec, upvec, fwdvec], axis = 2)

    def calculate_rot_mats(self):

        rrot = self.root_rot

        def _recurse(bone, rotation, c_idx):
            cIdx = self.bonelist.index(bone)

            if (pIdx < 0):
                #n_rot = c_rot
                new_pos = initial_position
            else:
                #n_rot = c_rot * rotations[pIdx]
                #new_pos = keyvector[pIdx] + n_rot.apply(self.tpose[cIdx] - self.tpose[pIdx])

                keyvector[cIdx] = new_pos

            for child in self.bonetree[bone]:
                _recurse(child, n_rot, cIdx)

            _recurse(self.root_idx, None, None)
        

# test_file = '../../data/h36m_zed/S7/S7_posing_2_zed34_test.npz'

# tfdata = np.load(test_file, allow_pickle = True)


fk = ForwardKinematics(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)
fktorch = ForwardKinematics_Torch(body_34_parts, body_34_tree, "PELVIS", body_34_tpose)

if (False):
    test_quats, test_kps, test_quant_quats = [tfdata[i] for i in ['quats', 'keypoints', 'quantized_quats']]

    rots = []
    for tq in test_quats:
        rots.append([Quaternion(u) for u in tq])

    #quant_torch = torch.unsqueeze(torch.tensor(test_quats), dim = 0).type(torch.Tensor)
    btpose_torch = torch.tensor(body_34_tpose)

    test_quat1_np = np.array([-0.1419, -0.0820, 0.400, 0.9018])
    test_quat2_np = np.array([ 0.27907279,  0.33488734, -0.44651646,  0.7814038 ])

    test_axis1 = test_quat1_np[:3] / np.linalg.norm(test_quat1_np[:3])
    test_theta1 = 2 * math.acos(test_quat1_np[3])

    test_axis2 = test_quat2_np[:3] / np.linalg.norm(test_quat2_np[:3])
    test_theta2 = 2 * math.acos(test_quat2_np[3])

    test_v_np = np.array([1.0, 2.0, 4.0000001])
    test_v_torch = torch.unsqueeze(torch.tensor(test_v_np), dim = 0)

    test_quat1_torch = torch.tensor(test_quat1_np)
    test_quat2_torch = torch.tensor(test_quat2_np)

    test_quat_torch = torch.reshape(torch.stack([test_quat1_torch, test_quat2_torch]), [1, 2, 1, 4])

    quats_torch = torch.unsqueeze(torch.tensor(test_quats), dim = 0).float()
#from zed_utilities import ForwardKinematics, ForwardKinematics_Torch, old_rotate_vector, batch_rotate_vector, test_v_np, test_v_torch, test_quat_torch, test_axis1, test_theta1, test_axis2, test_theta2, test_quat1_np, test_quat2_np, fktorch, quats_torch, Position, test_kps, body_34_parts, body_34_tree, fk

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_npz", type = str)

    parser.add_argument("in_npz", type = str)
    parser.add_argument("frame", type = int)
    parser.add_argument("out_csv", type = str)

    args = parser.parse_args()

    if (args.out_csv[-4:].lower() != ".csv"):
        print("Error: Not writing to a file that doesn't end in '.csv'")
        exit(0)
                        
    inf_data = np.load(args.in_npz, allow_pickle = True)
    npquat, np_kps = [inf_data[i] for i in ['quats', 'keypoints']]
    
    fkt = ForwardKinematics_Torch(body_34_parts, body_34_tree, 'PELVIS', body_34_tpose)
    fkn = ForwardKinematics(body_34_parts, body_34_tree, 'PELVIS', body_34_tpose)

    oframe = args.frame
    
    quats_torch = torch.unsqueeze(torch.tensor(npquat), dim = 0).float()[:, oframe:oframe + 1, :, :].cuda()
    print(quats_torch.shape)
    kp_torch = fkt.propagate(quats_torch).cuda()

    framelist = []
    # for frame in range(npquat.shape[0]):
    #     frot = [Quaternion(i) for i in npquat[frame]]
    #     framelist.append([j.np() for j in fk.propagate(frot, Position([0, 0, 0]))])


    frot = [Quaternion(i) for i in npquat[oframe]]
    framelist.append([j.np() for j in fk.propagate(frot, Position([0, 0, 0]))])
    recalc_kps = np.array(framelist)

    # Now npquat == quaternions [frame, bone, coord]
    # recalc_kps == keypoints using the old forward kinematics, calculated [frame, bone, coord]
    # np_kps = keypoints from the data file [frame, bone, coord]
    # kp_torch = keypoints calculated via torch batches - [batch, frame, bone, coord]

    header = ['Bone',
              'Qx', 'Qy', 'Qz', 'Qw',
              'Datafile_x', 'Datafile_y', 'Datafile_z',
              'Old_calc_x', 'Old_calc_y', 'Old_calc_z', 
              'Torch_calc_x', 'Torch_calc_y', 'Torch_calc_z']
    
    with open(args.out_csv, 'w') as ofp:

        ofp.write(",".join(header))
        ofp.write("\n")
        

        for i, p in enumerate(body_34_parts):
            line = []            
            line.append(p) # String
            line.append(str(float(npquat[oframe, i, 0])))
            line.append(str(float(npquat[oframe, i, 1])))
            line.append(str(float(npquat[oframe, i, 2])))
            line.append(str(float(npquat[oframe, i, 3])))           

            line.append(str(float(recalc_kps[0, i, 0])))
            line.append(str(float(recalc_kps[0, i, 1])))        
            line.append(str(float(recalc_kps[0, i, 2])))

            line.append(str(float(np_kps[oframe, i, 0])))
            line.append(str(float(np_kps[oframe, i, 1])))        
            line.append(str(float(np_kps[oframe, i, 2])))

            line.append(str(float(kp_torch[0, 0, i, 0])))
            line.append(str(float(kp_torch[0, 0, i, 1])))        
            line.append(str(float(kp_torch[0, 0, i, 2])))        

            ofp.write(",".join(line))
            ofp.write("\n")


