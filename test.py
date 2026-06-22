import os
import argparse

import torch
from torch.nn import DataParallel
import torch.nn.functional as F

from data.dataset import FSSDataset
from model.rsrm import RSRM
from common import utils
from common.evaluation import Evaluator
from common.logger import Logger, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Training-free Cross-domain Few-shot Segmentation via Robust Semantic Representation and Matching')

    parser.add_argument('--output-path',
                        type=str,
                        default='./output',
                        help='The path of output')
    
    parser.add_argument('--datapath',
                        type=str,
                        default="./datasets",
                        help='The path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='fss',
                        choices=['fss', 'deepglobe', 'isic', 'lung', 'pascal'],
                        help='validation dataset')
    parser.add_argument('--size',
                        type=int,
                        default=480,
                        help='Size of training samples')
    
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed to generate tesing samples')
    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        help='batch size of training')
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    
    parser.add_argument('--backbone',
                        type=str,
                        choices=['dinov3', 'dinov2', 'vit_b'],
                        default='dinov3',
                        help='backbone of semantic segmentation model')

    args = parser.parse_args()
    return args


def test(model, dataloader, args):
    utils.fix_randseed(args.seed)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)

        # [B,K,C,H,W] -> [K,B,C,H,W]
        img_s = batch['support_imgs'].permute(1,0,2,3,4)
        # [B,K,H,W] -> [K,B,H,W]
        mask_s = batch['support_masks'].permute(1,0,2,3)
        # [B,C,H,W]
        img_q = batch['query_img']
        # [B,H,W]
        mask_q = batch['query_mask']

        img_s_list = [img_s[k] for k in range(img_s.shape[0])]
        mask_s_list = [mask_s[k] for k in range(mask_s.shape[0])]

        # Model inference
        out_ls = model(img_s_list, mask_s_list, img_q)
        pred_mask = torch.argmax(out_ls[0], dim = 1)

        # Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), write_batch_idx=10)
    
    # Write evaluation results
    average_meter.write_result('Test')
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':
    args = parse_args()

    Logger.initialize(args)

    FSSDataset.initialize(img_size=args.size, datapath=args.datapath)
    testloader = FSSDataset.build_dataloader(args.dataset, 1, 0, 0, 'test', args.shot)

    model = RSRM(args.backbone)
    Logger.log_params(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DataParallel(model)
    model.to(device)
    model.eval()

    Evaluator.initialize()

    Logger.info('\n==================== Start Testing ====================')
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, testloader, args)
    Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')

