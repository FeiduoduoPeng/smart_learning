import sys
import os
import time

# POSE3D_DIR = os.getenv('POSE3D_DIR')
# if POSE3D_DIR == None:
#     print('please set the root dir of this package')
#     exit(-1)
# else:
#     sys.path.append(POSE3D_DIR+'/VideoPose3D/')

from common.arguments import parse_args
from common.h36m_dataset import h36m_skeleton
from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from common.utils import deterministic_random
from common.custom_dataset import CustomDataset

# import some common libraries
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import argparse
import sys
import os
import glob
import cv2
import rospy

# support c library
from ctypes import *

# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

####################################################################################

def parse_args_2d():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: mp4)',
        default='mp4',
        type=str
    )
    parser.add_argument(
        '--video',
        help='video (default: dance_0.mp4',
        default='dance_0.mp4',
        type=str
    )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()

def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    for line in pipe.stdout:
        w, h = line.decode().strip().split(',')
        return int(w), int(h)

def read_video(filename):
    w, h = get_resolution(filename)

    command = ['ffmpeg', '-i', filename, '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vsync', '0', '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    while True:
        data = pipe.stdout.read(w*h*3)
        if not data:
            break
        yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))

def preprocess2D(data):
    bb = data['boxes']
    kp = data['keypoints']
    metadata = data['metadata']
    results_bb = []
    results_kp = []
    for i in range(len(bb)):
        if len(bb[i][1]) == 0 or len(kp[i][1]) == 0:
            # No bbox/keypoints detected for this frame -> will be interpolated
            results_bb.append(np.full(4, np.nan, dtype=np.float32)) # 4 bounding box coordinates
            results_kp.append(np.full((17, 4), np.nan, dtype=np.float32)) # 17 COCO keypoints
            continue
        best_match = np.argmax(bb[i][1][:, 4])
        best_bb = bb[i][1][best_match, :4]
        best_kp = kp[i][1][best_match].T.copy()
        results_bb.append(best_bb)
        results_kp.append(best_kp)
        
    bb = np.array(results_bb, dtype=np.float32)
    kp = np.array(results_kp, dtype=np.float32)
    kp = kp[:, :, :2] # Extract (x, y)
    
    # Fix missing bboxes/keypoints by linear interpolation
    mask = ~np.isnan(bb[:, 0])
    indices = np.arange(len(bb))
    for i in range(4):
        bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])
    for i in range(17):
        for j in range(2):
            kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])
    
    print('{} total frames processed'.format(len(bb)))
    print('{} frames were interpolated'.format(np.sum(~mask)))
    print('----------')
    
    return {
        'start_frame': 0,    # Inclusive
        'end_frame': len(kp),    # Exclusive
        'bounding_boxes': bb,
        'keypoints': kp,
    }, metadata

def renderImg2D(bbox, kps, img):
    if len(bbox)==0 or len(kps)==0:
        return img
    img = cv2.rectangle(img, (bbox[0][0], bbox[0][1]),  (bbox[0][2], bbox[0][3]), (0,0,255), 4 ) # bouding box
    kps = kps[0]
    for i in range(17):
        img = cv2.circle(img, (kps[0][i], kps[1][i]), 2, (255,0,0), 2)    #keypoints
        # cv2.putText(img, str(i), (kps[0][i], kps[1][i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)

    pts = np.array(kps[:][0:2]).T
    lines = [
        np.array([pts[3], pts[1], pts[0], pts[2], pts[4]]).astype(np.int32).reshape((-1,1,2)),
        np.array([pts[0], pts[6], pts[8], pts[10]]).astype(np.int32).reshape((-1,1,2)),
        np.array([pts[0], pts[5], pts[7], pts[9]]).astype(np.int32).reshape((-1,1,2)),
        np.array([pts[0], pts[12], pts[14], pts[16]]).astype(np.int32).reshape((-1,1,2)),
        np.array([pts[0], pts[11], pts[13], pts[15]]).astype(np.int32).reshape((-1,1,2)),
    ]

    for item in lines:
        img = cv2.polylines(img, [item], False, (255,0,0), 2)
    return img

def infer2D(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)
    predictor = DefaultPredictor(cfg)

    video_name = args.video

    # out_name = os.path.join( args.output_dir, os.path.basename(video_name) )
    print('Processing {}'.format(video_name))

    boxes, segments, keypoints = [], [], []

    isFirstFrm = True

    for frame_i, im in enumerate(read_video(video_name)):
        t = time.time()

        if isFirstFrm:
            isFirstFrm = False
            out_vdo = cv2.VideoWriter('rendering.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 10, (im.shape[1],im.shape[0]))

        outputs = predictor(im)['instances'].to('cpu')
        # print('Frame {} processed in {:.3f}s'.format(frame_i, time.time() - t))

        has_bbox = False
        if outputs.has('pred_boxes'):
            bbox_tensor = outputs.pred_boxes.tensor.numpy()
            if len(bbox_tensor) > 0:
                has_bbox = True
                scores = outputs.scores.numpy()[:, None]
                bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1) # coordinator bounding box

        if has_bbox:
            kps = outputs.pred_keypoints.numpy()
            kps_xy = kps[:, :, :2]
            kps_prob = kps[:, :, 2:3]
            kps_logit = np.zeros_like(kps_prob) # Dummy
            kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
            kps = kps.transpose(0, 2, 1)

        else:
            kps = []
            bbox_tensor = []

        img = renderImg2D(bbox_tensor, kps, im)
        out_vdo.write(img)
            
        # Mimic Detectron1 format
        cls_boxes = [[], bbox_tensor]
        cls_keyps = [[], kps]
        
        boxes.append(cls_boxes)
        segments.append(None)
        keypoints.append(cls_keyps)

        # cv2.imwrite('show.jpg', img)

        metadata = { 'w': im.shape[1], 'h': im.shape[0],}
        # print(metadata)
    out_vdo.release()
    return  {'boxes': boxes, 'keypoints':keypoints, 'metadata': metadata}

useCausal = True
def load3DModel():
    filter_widths = [3, 3, 3, 3, 3]
    #__init__(self, num_joints_in, in_features, num_joints_out, filter_widths, causal=False, dropout=0.25, channels=1024)
    # TemporalModelOptimized1f
    model_pos = TemporalModel(17, 2, 17, filter_widths=filter_widths, causal=useCausal, dropout=0.0, channels=1024, dense=False)
    # model_pos = TemporalModelOptimized1f(17, 2, 17, filter_widths=filter_widths, causal=useCausal, dropout=0.0, channels=1024)
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
        
    # chk_filename =  'checkpoint/pretrained_h36m_cpn.bin'
    chk_filename =  'checkpoint/pretrained_h36m_detectron_coco.bin'
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)

    print('Loading checkpoint', chk_filename)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    if 'model_traj' in checkpoint:
        model_pos.load_state_dict(checkpoint['model_traj'])
    else:
        model_pos.load_state_dict(checkpoint['model_pos'])

    return model_pos

def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]
    
def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    # Reverse camera frame normalization
    return (X + [1, h/w])*w/2

def infer3D(model3d, kp2d, imgSize):
    receptive_field = model3d.receptive_field()
    pad = (receptive_field - 1) // 2 # Padding on each side
    if useCausal:
        causal_shift = pad
    else:
        causal_shift = 0

    # kp2d = np.array([kp2d])
    kp2d = normalize_screen_coordinates(kp2d, imgSize[1], imgSize[0])

    #bypass the data-generator to speed up this algorithm
    with torch.no_grad():
        model3d.eval()
        seq_2d = kp2d
        batch_2d = np.expand_dims( np.pad(seq_2d, ((pad + causal_shift, pad - causal_shift), (0, 0), (0, 0)), 'edge'), axis=0 )
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()

        predicted_3d_pos = model3d(inputs_2d)
        return predicted_3d_pos.squeeze(0).cpu().numpy()

def normalizeSkeleton(data_3d):
    pass

def plot3D(data_3d):
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(10,-80)

    for i, frame in enumerate(data_3d):
        plt.cla()
        ax.plot3D([-1,1],[0,0],[0,0], c = 'r')
        ax.plot3D([0,0],[-1,1],[0,0], c = 'g')
        ax.plot3D([0,0],[0,0],[-1,1], c = 'b')

        x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]
        y, z = z, -y    #rotate -pi/2 along the x axis
        leg = [
            [x[3], x[2], x[1], x[0], x[4], x[5], x[6]],
            [y[3], y[2], y[1], y[0], y[4], y[5], y[6]],
            [z[3], z[2], z[1], z[0], z[4], z[5], z[6]],
        ]
        spine = [
            [x[0], x[7], x[8], x[9], x[10]], 
            [y[0], y[7], y[8], y[9], y[10]], 
            [z[0], z[7], z[8], z[9], z[10]],
        ]
        leftArm = [ 
            [x[8], x[11], x[12], x[13]],
            [y[8], y[11], y[12], y[13]],
            [z[8], z[11], z[12], z[13]],
        ]
        rightArm = [ 
            [x[8], x[14], x[15], x[16]],
            [y[8], y[14], y[15], y[16]],
            [z[8], z[14], z[15], z[16]],
        ]

        ax.plot3D(leg[0], leg[1], leg[2], c='b')
        ax.plot3D(spine[0], spine[1], spine[2], c='b')
        ax.plot3D(rightArm[0], rightArm[1], rightArm[2], c='r')
        ax.plot3D(leftArm[0], leftArm[1], leftArm[2], c='g')

        plt.draw()
        plt.savefig('tmp/'+str(i)+'.jpg')
        # print('---')
        plt.pause(0.001)
        input('press enter to continue')

def ske2angs(data_3d):
    s2a = cdll.LoadLibrary('./libske2ang.so')
    dblPtr = POINTER(c_double)
    
    text_file = open('data.txt', 'w')
    for ske in data_3d:
        ske= [[-item[1], -item[0], -item[2]] for item in ske]    #rotate points
        flatPts = np.array(ske, dtype=c_double).flatten()
        skePtr = flatPts.ctypes.data_as(dblPtr)

        angsOut = np.zeros(21, dtype=c_double)
        angsPtr = angsOut.ctypes.data_as(dblPtr)

        s2a.skeleton2angles(skePtr,angsPtr)
        res = []
        for i in range(20):
            res.append(angsPtr[i])
            text_file.write('%s,'%angsPtr[i])
        res.append(angsPtr[20])
        text_file.write('%s\n'%angsPtr[20])
        text_file.flush()
        # print(res)
        # print('----------')




def main():
    setup_logger()
    # rospy.init_node('smart_learning')
    args_2d = parse_args_2d()
    data_2d = infer2D(args_2d)
    data, meta= preprocess2D(data_2d)
    model_3d = load3DModel()
    # print(data['keypoints'][0])

    data_3d = infer3D( model_3d, data['keypoints'], (meta['w'], meta['h']) )

    ske2angs(data_3d)
    # plot3D(data_3d)
    # print(data_3d)


#########################################################################################
#################################  Main Function  #######################################
#########################################################################################
if __name__ == '__main__':
    main()