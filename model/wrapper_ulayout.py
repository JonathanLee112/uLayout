import pdb
import time
import glob
import json
import os
import sys
import yaml
import pathlib
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from loguru import logger
from datasets.mp3d_hn_ori_dataset import PanoCorBonDataset
from datasets.pano_st2d3d_dataset import PanoSt2D3DDataset
from datasets.lsun_preproc_dataset import LSUNPreprocDataset
from utils.pp_utils import read_image
from utils.io_utils import save_json_dict, print_cfg_information, create_directory, save_cfg, plotXY, prepare_corner_wo_occlusion
from utils.eval_utils_research import   eval_2d3d_iuo_from_tensors, compute_L1_loss, compute_L1_loss_range, \
                                        find_predict_range, compute_normal_gradient_loss, phi_coords2xyz
from utils.pp_utils import boundary2depth_ceilingfloor_torch, pp_map2boundary, gen_pp_boundary_map

from imageio import imread, imwrite

class WrapperuLayout:
    def __init__(self, cfg):
        self.cfg = cfg
        from model.Swim_Transformer.lgt_net import LGT_Net

        # ! Setting cuda-device
        self.device = torch.device(
            f"cuda:{cfg.cuda_device}" if torch.cuda.is_available() else "cpu"
        )

        logger.info("Loading uLayout...")
        self.net = LGT_Net(output_name='Horizon', decoder_name='SWG_Transformer', win_size=16, depth=8, dropout=0.0, ape='lr_parameter').to(self.device)
        if self.cfg.mode == "val" or self.cfg.mode == "test":
            assert os.path.isfile(cfg.ckpt_dir), f"Not found {cfg.ckpt_dir}"
            self.net = self.load_trained_model(self.net, cfg.ckpt_dir)
            logger.info(f"loading check point from: {cfg.ckpt_dir}")
        logger.info("uLayout Wrapper Successfully initialized")

        self.current_epoch = 0
        self.tb_writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, 'log'))

    def load_trained_model(self, Net, path):
        state_dict = torch.load(path, map_location='cpu')
        Net.load_state_dict(state_dict['state_dict'])
        return Net

    def set_optimizer(self):
        if self.cfg.model.optimizer == "SGD":
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.cfg.model.lr,
                momentum=self.cfg.model.beta1,
                weight_decay=self.cfg.model.weight_decay,
            )
        elif self.cfg.model.optimizer == "Adam":
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.cfg.model.lr,
                betas=(self.cfg.model.beta1, 0.999),
                weight_decay=self.cfg.model.weight_decay,
            )
        else:
            raise NotImplementedError()

    def set_scheduler(self):
        decayRate = self.cfg.model.lr_decay_rate
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=decayRate
        )

    def train_loop(self):
        if not self.is_training:
            logger.warning("Wrapper is not ready for training")
            return False

        print_cfg_information(self.cfg)

        self.net.train()

        iterator_train = iter(self.train_loader)
        for _ in trange(len(self.train_loader),
                        desc=f"Training HorizonNet epoch:{self.current_epoch}/{self.cfg.model.epochs}"):

            self.iterations += 1
            cur_lr = self.cfg.model.lr
            sample = next(iterator_train)
            x = sample['img']
            y_bon_ref = sample['boundary']
            y_bon_ref_d = sample['depth']
            eval_range = sample['eval_range']
            gt_type = sample['gt_type']

            y_bon_est, _ = self.net(x.to(self.device))
            eval_range = eval_range.int()
            gt_type = gt_type.int()
            if y_bon_est is np.nan:
                raise ValueError("Nan value")

            loss = 0
            for i in range(y_bon_est.shape[0]):
                # new loss that only evaluate the range that has ground truth
                eval_c_idx = np.where(eval_range[i,0] != 0)[0]
                eval_f_idx = np.where(eval_range[i,1] != 0)[0]
                if len(eval_c_idx) == y_bon_est[i].shape[-1] and len(eval_f_idx) == y_bon_est[i].shape[-1]:
                    loss += compute_L1_loss(y_bon_est[i].to(self.device),
                                            y_bon_ref[i].to(self.device))
                    
                    y_bon_est_d = boundary2depth_ceilingfloor_torch(y_bon_est[i].to(self.device), gt_type, eval_range[i])
                    # depth loss, coefficient 0.1
                    loss += 0.1 * compute_L1_loss(y_bon_est_d,
                                                y_bon_ref_d[i].to(self.device))
                    # normal and gradient loss, coefficient 0.1
                    loss += 0.1 * compute_normal_gradient_loss(y_bon_est_d[0],
                                            y_bon_ref_d[i,0].to(self.device))
                    loss += 0.1 * compute_normal_gradient_loss(y_bon_est_d[1],
                                            y_bon_ref_d[i,1].to(self.device))
                else:
                    ##### if only solely train perspective data, set eval_range to 1 #####
                    ##### please uncomment the following line ######
                    # eval_range[i] = torch.ones_like(eval_range[i]) 
                    ######################################################################
                    loss += compute_L1_loss_range(y_bon_est[i].to(self.device),
                                                    y_bon_ref[i].to(self.device),
                                                    eval_range[i].to(self.device))
                    
            if np.isnan(loss.item()):
                raise ValueError("something is wrong")
            self.tb_writer.add_scalar(
                "train/loss", loss.item(), self.iterations)
            self.tb_writer.add_scalar(
                "train/lr", cur_lr, self.iterations)

            # back-prop
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(
            #     self.net.parameters(), 3.0, norm_type="inf")
            self.optimizer.step()

        # Epoch finished
        self.current_epoch += 1

        # ! Saving model
        if self.cfg.model.get("save_every") > 0:
            if self.current_epoch % self.cfg.model.get("save_every", 5) == 0:
                self.save_model(f"model_at_{self.current_epoch}.pth")

        if self.current_epoch > self.cfg.model.epochs:
            self.is_training = False


        return self.is_training

    def save_current_scores(self):
        # ! Saving current epoch data
        fn = os.path.join(self.dir_ckpt, f"valid_eval_{self.current_epoch}.json")
        save_json_dict(filename=fn, dict_data=self.curr_scores)
        # ! Save the best scores in a json file regardless of saving the model or not
        save_json_dict(
            dict_data=self.best_scores,
            filename=os.path.join(self.dir_ckpt, "best_score.json")
        )

    def valid_iou_loop(self, only_val=False):
        print_cfg_information(self.cfg)
        self.net.eval()
        iterator_valid_iou = iter(self.valid_iou_loader)
        total_eval = {}
        invalid_cnt = 0
        pano_ceilling_num = 0
        pano_floor_num = 0
        pp_ceilling_num = 0
        pp_floor_num = 0

        for _ in trange(len(iterator_valid_iou), desc="IoU Validation epoch %d" % self.current_epoch):
            #x, y_bon_ref, std, u_range, eval_range = next(iterator_valid_iou)
            sample = next(iterator_valid_iou)
            x = sample['img']
            # name = sample['img_name']
            y_bon_ref = sample['boundary']
            # gt_corner = sample['corner']
            # gt_ratio = sample['ratio']
            u_range = sample['u_range']
            eval_range = sample['eval_range']
            gt_type = sample['gt_type']
            
            u_range = u_range.int()
            eval_range = eval_range.int() 
            gt_type = gt_type.int()

            with torch.no_grad():
                y_bon_est, _ = self.net(x.to(self.device))

                true_eval = {"2DIoU_pano": [], "3DIoU_pano": [], "2DIoU_pp_floor": [], "2DIoU_pp_ceiling": []}
                for img, gt, est, gt_range, img_range, img_type in zip(x.cpu().numpy() , y_bon_ref.cpu().numpy(), y_bon_est.cpu().numpy(), eval_range.cpu().numpy(),
                                                                  u_range.cpu().numpy(), gt_type.cpu().numpy()):

                    # img_type 0: pano, 1: pp (ceiling and floor), 2: pp (ceiling), 3: pp (floor)
                    pred_range, est = find_predict_range(img, gt, est, img_range, img_type-1)
                    eval_2d3d_iuo_from_tensors(est[None], gt[None], true_eval, gt_range, pred_range, img_range, 1.6)

                loss = 0
                for i in range(y_bon_est.shape[0]):
                    # new loss that only evaluate the range that has ground truth
                    eval_c_idx = np.where(eval_range[i,0] != 0)[0]
                    eval_f_idx = np.where(eval_range[i,1] != 0)[0]
                    if len(eval_c_idx) == y_bon_est[i].shape[-1] and len(eval_f_idx) == y_bon_est[i].shape[-1]:
                        loss += compute_L1_loss(y_bon_est[i].to(self.device),
                                                y_bon_ref[i].to(self.device))
                    else:
                        loss += compute_L1_loss_range(y_bon_est[i].to(self.device),
                                                      y_bon_ref[i].to(self.device),
                                                      eval_range[i])
                        
                local_eval = dict(
                    loss=loss,
                )
                local_eval["2DIoU_pano"] = torch.FloatTensor(
                    [true_eval["2DIoU_pano"]]).sum()
                pano_floor_num += len(true_eval["2DIoU_pano"])
                local_eval["3DIoU_pano"] = torch.FloatTensor(
                    [true_eval["3DIoU_pano"]]).sum()
                pano_ceilling_num += len(true_eval["3DIoU_pano"])
                local_eval["2DIoU_pp_floor"] = torch.FloatTensor(
                    [true_eval["2DIoU_pp_floor"]]).sum()
                pp_floor_num += len(true_eval["2DIoU_pp_floor"])
                local_eval["2DIoU_pp_ceiling"] = torch.FloatTensor(
                    [true_eval["2DIoU_pp_ceiling"]]).sum()
                pp_ceilling_num += len(true_eval["2DIoU_pp_ceiling"])
            try:
                for k, v in local_eval.items():
                    if v.isnan():
                        continue
                    total_eval[k] = total_eval.get(k, 0) + v.item()
            except:
                invalid_cnt += 1
                pass

        if only_val:
            if "3DIoU_pano" not in total_eval:
                total_eval["3DIoU_pano"] = 0
            if "2DIoU_pano" not in total_eval:
                total_eval["2DIoU_pano"] = 0
            if "2DIoU_pp_floor" not in total_eval:
                total_eval["2DIoU_pp_floor"] = 0
            if "2DIoU_pp_ceiling" not in total_eval:
                total_eval["2DIoU_pp_ceiling"] = 0
            curr_score_3d_iou_pano = total_eval["3DIoU_pano"] / pano_ceilling_num if pano_ceilling_num != 0 else 0
            curr_score_2d_iou_pano = total_eval["2DIoU_pano"] / pano_floor_num if pano_floor_num != 0 else 0
            curr_score_2d_iou_pp_floor = total_eval["2DIoU_pp_floor"] / pp_floor_num if pp_floor_num != 0 else 0
            curr_score_2d_iou_pp_ceiling = total_eval["2DIoU_pp_ceiling"] / pp_ceilling_num if pp_ceilling_num != 0 else 0
            curr_score_avg_2d_iou_pp = (total_eval["2DIoU_pp_floor"] + total_eval["2DIoU_pp_ceiling"]) / (pp_floor_num + pp_ceilling_num) if (pp_floor_num + pp_ceilling_num) != 0 else 0
            curr_score_total_iou = (total_eval["3DIoU_pano"] + total_eval["2DIoU_pp_floor"] + total_eval["2DIoU_pp_ceiling"]) / (pano_ceilling_num + pp_floor_num + pp_ceilling_num) if (pano_ceilling_num + pp_floor_num + pp_ceilling_num) != 0 else 0
            logger.info(f"Pano 3D-IoU score(ceiling 2Diou in pp): {curr_score_3d_iou_pano:.4f}")
            logger.info(f"Pano 2D-IoU score(floor 2Diou in pp): {curr_score_2d_iou_pano:.4f}")
            logger.info(f"PP 2D-IoU score(floor): {curr_score_2d_iou_pp_floor:.4f}")
            logger.info(f"PP 2D-IoU score(ceiling): {curr_score_2d_iou_pp_ceiling:.4f}")
            logger.info(f"PP 2D-IoU score(avg): {curr_score_avg_2d_iou_pp:.4f}")
            logger.info(f"Total IoU score: {curr_score_total_iou:.4f}")
            return {"2D-IoU_pano": curr_score_2d_iou_pano, "3D-IoU_pano": curr_score_3d_iou_pano, \
                    "2D-IoU_pp_floor": curr_score_2d_iou_pp_floor, "2D-IoU_pp_ceiling": curr_score_2d_iou_pp_ceiling}

        self.tb_writer.add_scalar(
            "valid_IoU/loss", total_eval["loss"] / len(iterator_valid_iou), self.current_epoch)
        self.tb_writer.add_scalar(
            "valid_IoU/Pano_2DIoU", total_eval["2DIoU_pano"] / pano_floor_num, self.current_epoch)
        self.tb_writer.add_scalar(
            "valid_IoU/Pano_3DIoU", total_eval["3DIoU_pano"] / pano_ceilling_num, self.current_epoch)
        self.tb_writer.add_scalar(
            "valid_IoU/PP_2DIoU_floor", total_eval["2DIoU_pp_floor"] / pp_floor_num, self.current_epoch)
        self.tb_writer.add_scalar(
            "valid_IoU/PP_2DIoU_ceiling", total_eval["2DIoU_pp_ceiling"] / pp_ceilling_num, self.current_epoch)
        self.tb_writer.add_scalar(
            "valid_IoU/PP_2DIoU_avg", (total_eval["2DIoU_pp_floor"] + total_eval["2DIoU_pp_ceiling"]) / (pp_floor_num + pp_ceilling_num), self.current_epoch)
        self.tb_writer.add_scalar(
            "valid_IoU/Total_IoU", (total_eval["3DIoU_pano"] + total_eval["2DIoU_pp_floor"] + total_eval["2DIoU_pp_ceiling"]) / (pano_ceilling_num + pp_floor_num + pp_ceilling_num), self.current_epoch)

        # Save best validation loss model
        if "3DIoU_pano" not in total_eval:
            total_eval["3DIoU_pano"] = 0
        if "2DIoU_pano" not in total_eval:
            total_eval["2DIoU_pano"] = 0
        if "2DIoU_pp_floor" not in total_eval:
            total_eval["2DIoU_pp_floor"] = 0
        if "2DIoU_pp_ceiling" not in total_eval:
            total_eval["2DIoU_pp_ceiling"] = 0
        curr_score_3d_iou_pano = total_eval["3DIoU_pano"] / pano_ceilling_num if pano_ceilling_num != 0 else 0
        curr_score_2d_iou_pano = total_eval["2DIoU_pano"] / pano_floor_num if pano_floor_num != 0 else 0
        curr_score_2d_iou_pp_floor = total_eval["2DIoU_pp_floor"] / pp_floor_num if pp_floor_num != 0 else 0
        curr_score_2d_iou_pp_ceiling = total_eval["2DIoU_pp_ceiling"] / pp_ceilling_num if pp_ceilling_num != 0 else 0
        curr_score_avg_2d_iou_pp = (total_eval["2DIoU_pp_floor"] + total_eval["2DIoU_pp_ceiling"]) / (pp_floor_num + pp_ceilling_num) if (pp_floor_num + pp_ceilling_num) != 0 else 0
        curr_score_total_iou = (total_eval["3DIoU_pano"] + total_eval["2DIoU_pp_floor"] + total_eval["2DIoU_pp_ceiling"]) / (pano_ceilling_num + pp_floor_num + pp_ceilling_num) if (pano_ceilling_num + pp_floor_num + pp_ceilling_num) != 0 else 0
        # ! Saving current score
        self.curr_scores['pano_iou_valid_scores'] = dict(
            best_3d_pano_iou_score=curr_score_3d_iou_pano,
            best_2d_pano_iou_score=curr_score_2d_iou_pano
        )
        self.curr_scores['pp_iou_valid_scores'] = dict(
            best_2d_iou_floor_score=curr_score_2d_iou_pp_floor,
            best_2d_iou_ceiling_score=curr_score_2d_iou_pp_ceiling,
            best_2d_iou_avg_score=curr_score_avg_2d_iou_pp
        )
        self.curr_scores['total_iou_valid_scores'] = dict(
            best_total_iou_score=curr_score_total_iou
        )
        # save current pano iou score
        if self.best_scores.get("pano_best_iou_valid_score") is None:
            logger.info(f"Pano Best 3D-IoU score: {curr_score_3d_iou_pano:.4f}")
            logger.info(f"Pano Best 2D-IoU score: {curr_score_2d_iou_pano:.4f}")
            self.best_scores["pano_best_iou_valid_score"] = dict(
                best_3d_pano_iou_score=curr_score_3d_iou_pano,
                best_2d_pano_iou_score=curr_score_2d_iou_pano
            )
        else:
            best_3d_pano_iou_score = self.best_scores["pano_best_iou_valid_score"]['best_3d_pano_iou_score']
            best_2d_pano_iou_score = self.best_scores["pano_best_iou_valid_score"]['best_2d_pano_iou_score']

            logger.info(
                f"3D-IoU: Best: {best_3d_pano_iou_score:.4f} vs Curr:{curr_score_3d_iou_pano:.4f}")
            logger.info(
                f"2D-IoU: Best: {best_2d_pano_iou_score:.4f} vs Curr:{curr_score_2d_iou_pano:.4f}")

            if best_3d_pano_iou_score < curr_score_3d_iou_pano:
                logger.info(
                    f"New Pano 3D-IoU Best Score {curr_score_3d_iou_pano: 0.4f}")
                self.best_scores["pano_best_iou_valid_score"]['best_3d_pano_iou_score'] = curr_score_3d_iou_pano
                self.save_model("best_3d_pano_iou_valid.pth")

            if best_2d_pano_iou_score < curr_score_2d_iou_pano:
                logger.info(
                    f"New Pano 2D-IoU Best Score {curr_score_2d_iou_pano: 0.4f}")
                self.best_scores["pano_best_iou_valid_score"]['best_2d_pano_iou_score'] = curr_score_2d_iou_pano
                self.save_model("best_2d_pano_iou_valid.pth")
        
        # save current pp iou score
        if self.best_scores.get("pp_best_iou_valid_score") is None:
            logger.info(f"PP Best 2D-IoU floor score: {curr_score_2d_iou_pp_floor:.4f}")
            logger.info(f"PP Best 2D-IoU ceiling score: {curr_score_2d_iou_pp_ceiling:.4f}")
            logger.info(f"PP Best 2D-IoU avg score: {curr_score_avg_2d_iou_pp:.4f}")
            self.best_scores["pp_best_iou_valid_score"] = dict(
                best_2d_iou_floor_score=curr_score_2d_iou_pp_floor,
                best_2d_iou_ceiling_score=curr_score_2d_iou_pp_ceiling,
                best_2d_iou_avg_score=curr_score_avg_2d_iou_pp
            )
        else:
            best_2d_pp_floor_iou_score = self.best_scores["pp_best_iou_valid_score"]['best_2d_iou_floor_score']
            best_2d_pp_ceiling_iou_score = self.best_scores["pp_best_iou_valid_score"]['best_2d_iou_ceiling_score']
            best_2d_pp_avg_iou_score = self.best_scores["pp_best_iou_valid_score"]['best_2d_iou_avg_score']

            logger.info(
                f"PP 2D-IoU floor: Best: {best_2d_pp_floor_iou_score:.4f} vs Curr:{curr_score_2d_iou_pp_floor:.4f}")
            logger.info(
                f"PP 2D-IoU ceiling: Best: {best_2d_pp_ceiling_iou_score:.4f} vs Curr:{curr_score_2d_iou_pp_ceiling:.4f}")
            logger.info(
                f"PP 2D-IoU avg: Best: {best_2d_pp_avg_iou_score:.4f} vs Curr:{curr_score_avg_2d_iou_pp:.4f}")

            if best_2d_pp_floor_iou_score < curr_score_2d_iou_pp_floor:
                logger.info(
                    f"New PP 2D-IoU floor Best Score {curr_score_2d_iou_pp_floor: 0.4f}")
                self.best_scores["pp_best_iou_valid_score"]['best_2d_iou_floor_score'] = curr_score_2d_iou_pp_floor
                self.save_model("best_2d_pp_floor_iou_valid.pth")

            if best_2d_pp_ceiling_iou_score < curr_score_2d_iou_pp_ceiling:
                logger.info(
                    f"New PP 2D-IoU ceiling Best Score {curr_score_2d_iou_pp_ceiling: 0.4f}")
                self.best_scores["pp_best_iou_valid_score"]['best_2d_iou_ceiling_score'] = curr_score_2d_iou_pp_ceiling
                self.save_model("best_2d_pp_ceiling_iou_valid.pth")

            if best_2d_pp_avg_iou_score < curr_score_avg_2d_iou_pp:
                logger.info(
                    f"New PP 2D-IoU avg Best Score {curr_score_avg_2d_iou_pp: 0.4f}")
                self.best_scores["pp_best_iou_valid_score"]['best_2d_iou_avg_score'] = curr_score_avg_2d_iou_pp
                self.save_model("best_2d_pp_avg_iou_valid.pth")
        
        # save current total iou score
        if self.best_scores.get("total_best_iou_valid_score") is None:
            logger.info(f"Total Best IoU score: {curr_score_total_iou:.4f}")
            self.best_scores["total_best_iou_valid_score"] = dict(
                best_total_iou_score=curr_score_total_iou
            )
        else:
            best_total_iou_score = self.best_scores["total_best_iou_valid_score"]['best_total_iou_score']

            logger.info(
                f"Total IoU: Best: {best_total_iou_score:.4f} vs Curr:{curr_score_total_iou:.4f}")

            if best_total_iou_score < curr_score_total_iou:
                logger.info(
                    f"New Total IoU Best Score {curr_score_total_iou: 0.4f}")
                self.best_scores["total_best_iou_valid_score"]['best_total_iou_score'] = curr_score_total_iou
                self.save_model("best_total_iou_valid.pth")

    def save_model(self, filename):
        if not self.cfg.model.get("save_ckpt", True):
            return
        # ! Saving the current model
        state_dict = OrderedDict(
            {
                "args": self.cfg,
                "state_dict": self.net.state_dict(),
            }
        )
        torch.save(state_dict, os.path.join(
            self.dir_ckpt, filename))

    def prepare_for_training_multi_dataset_mp3d(self, mp3d_dir):
        self.is_training = True
        self.current_epoch = 0
        self.iterations = 0
        self.best_scores = dict()
        self.curr_scores = dict()
        self.set_optimizer()
        #self.set_scheduler()
        self.set_multi_train_dataloader_mp3d(mp3d_dir)
        self.set_log_dir()
        save_cfg(os.path.join(self.dir_ckpt, 'mp3d_cfg.yaml'), self.cfg)

    def prepare_for_training_multi_dataset_panost2d3d(self, pano_st2d3d_dir, subset='pano'):
        self.is_training = True
        self.current_epoch = 0
        self.iterations = 0
        self.best_scores = dict()
        self.curr_scores = dict()
        self.set_optimizer()
        #self.set_scheduler()
        self.set_multi_train_dataloader_panost2d3d(pano_st2d3d_dir, subset)
        self.set_log_dir()
        save_cfg(os.path.join(self.dir_ckpt, 'pano_st2d3d_cfg.yaml'), self.cfg)

    def set_log_dir(self):
        output_dir = os.path.join(self.cfg.output_dir, self.cfg.id_exp)
        create_directory(output_dir, delete_prev=False)
        logger.info(f"Output directory: {output_dir}")
        self.dir_log = os.path.join(output_dir, 'log')
        self.dir_ckpt = os.path.join(output_dir, 'ckpt')
        os.makedirs(self.dir_log, exist_ok=True)
        os.makedirs(self.dir_ckpt, exist_ok=True)

        self.tb_writer = SummaryWriter(log_dir=self.dir_log)

    def prepare_for_validation_multi_dataset(self):
        self.is_training = False
        self.current_epoch = 0
        self.iterations = 0
        self.best_scores = dict()
        self.curr_scores = dict()

    def set_multi_train_dataloader_mp3d(self, mp3d_dir):
        logger.info("Setting Training Dataloader")
        ######### mp3d dataset #########
        mp3d_dataset = PanoCorBonDataset(mp3d_dir, 'train')
        logger.info(f'mp3d_dataset_train: {len(mp3d_dataset)}')
        ####### lsun dataset #########
        lsun_dataset = LSUNPreprocDataset(self.cfg.lsun.data_dir, 'train', self.cfg.lsun.fix_shape)
        logger.info(f'lsun_dataset_train: {len(lsun_dataset)}')
        ######### concat dataset #########
        concat_dataset = ConcatDataset([mp3d_dataset, lsun_dataset])
        logger.info(f'concat_dataset_train: {len(concat_dataset)}')

        self.train_loader = DataLoader(
            concat_dataset,
            batch_size=self.cfg.model.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.model.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed(self.cfg.model.seed),
        )

    def set_multi_train_dataloader_panost2d3d(self, pano_st2d3d_dir, subset='pano'):
        logger.info("Setting Training Dataloader")
        assert subset in ['pano', 'st2d3d'], 'subset should be either pano or st2d3d'
        if subset == 'pano':
            pano_st2d3d_dataset = ConcatDataset([
                PanoSt2D3DDataset(pano_st2d3d_dir, mode='train', subset='pano'),
                PanoSt2D3DDataset(pano_st2d3d_dir, mode='train', subset='st2d3d'),
                PanoSt2D3DDataset(pano_st2d3d_dir, mode='val', subset='st2d3d', flip=True, rotate=True, gamma=True, stretch=True),
                PanoSt2D3DDataset(pano_st2d3d_dir, mode='test', subset='st2d3d', flip=True, rotate=True, gamma=True, stretch=True)
            ])
            logger.info(f'pano + whole st2d3d dataset train: {len(pano_st2d3d_dataset)}')
        elif subset == 'st2d3d':
            pano_st2d3d_dataset = ConcatDataset([
                PanoSt2D3DDataset(pano_st2d3d_dir, mode='train', subset='st2d3d'),
                PanoSt2D3DDataset(pano_st2d3d_dir, mode='train', subset='pano'),
                PanoSt2D3DDataset(pano_st2d3d_dir, mode='val', subset='pano', flip=True, rotate=True, gamma=True, stretch=True),
                PanoSt2D3DDataset(pano_st2d3d_dir, mode='test', subset='pano', flip=True, rotate=True, gamma=True, stretch=True)
            ])
            logger.info(f'st2d3d dataset train: {len(pano_st2d3d_dataset)}')
        lsun_dataset = LSUNPreprocDataset(self.cfg.lsun.data_dir, 'train', self.cfg.lsun.fix_shape)
        logger.info(f'lsun_dataset_train: {len(lsun_dataset)}')
        concat_dataset = ConcatDataset([pano_st2d3d_dataset, lsun_dataset])
        logger.info(f'concat_dataset_train: {len(concat_dataset)}')

        self.train_loader = DataLoader(
            concat_dataset,
            batch_size=self.cfg.model.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.model.num_workers,
            #num_workers=0,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed(self.cfg.model.seed),
        )
    
    def set_multi_valid_dataloader_mp3d(self, mp3d_dir, mode='val'):
        logger.info(f"Setting IoU {mode} Dataloader")
        mp3d_dataset = PanoCorBonDataset(mp3d_dir, mode)
        logger.info(f'mp3d_dataset_{mode}: {len(mp3d_dataset)}')
        lsun_dataset = LSUNPreprocDataset(self.cfg.lsun.data_dir, 'val', self.cfg.lsun.fix_shape)
        logger.info(f'lsun_dataset_{mode}: {len(lsun_dataset)}')
        concat_dataset = ConcatDataset([mp3d_dataset, lsun_dataset])
        logger.info(f'concat_dataset_{mode}: {len(concat_dataset)}')

        self.valid_iou_loader = DataLoader(
            concat_dataset,
            batch_size=self.cfg.model.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.model.num_workers,
            # num_workers=0,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed(self.cfg.model.seed)
        )

    def set_multi_valid_dataloader_panost2d3d(self, pano_st2d3d_dir, subset='pano', mode='val'):
        logger.info(f"Setting IoU {mode} Dataloader")
        pano_st2d3d_dataset = PanoSt2D3DDataset(pano_st2d3d_dir, mode, subset=subset)
        logger.info(f'pano_st2d3d_dataset_{mode}: {len(pano_st2d3d_dataset)}')
        lsun_dataset = LSUNPreprocDataset(self.cfg.lsun.data_dir, 'val', self.cfg.lsun.fix_shape)
        logger.info(f'lsun_dataset_{mode}: {len(lsun_dataset)}')
        concat_dataset = ConcatDataset([pano_st2d3d_dataset, lsun_dataset])
        logger.info(f'concat_dataset_{mode}: {len(concat_dataset)}')

        self.valid_iou_loader = DataLoader(
            concat_dataset,
            batch_size=self.cfg.model.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.model.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed(self.cfg.model.seed)
        )

    def set_valid_dataloader_mp3d(self, mp3d_dir, mode='val'):
        logger.info(f"Setting IoU {mode} Dataloader")
        mp3d_dataset = PanoCorBonDataset(mp3d_dir, mode)
        logger.info(f'mp3d_dataset_{mode}: {len(mp3d_dataset)}')

        self.valid_iou_loader = DataLoader(
            mp3d_dataset,
            batch_size=self.cfg.model.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.model.num_workers,
            # num_workers=0,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed(self.cfg.model.seed)
        )

    def set_valid_dataloader_panost2d3d(self, pano_st2d3d_dir, subset='pano', mode='val'):
        assert subset in ['pano', 'st2d3d'], 'subset should be either pano or st2d3d'
        logger.info(f"Setting IoU {mode} Dataloader")
        pano_st2d3d_dataset = PanoSt2D3DDataset(pano_st2d3d_dir, mode, subset=subset)
        logger.info(f'pano_st2d3d_dataset_{mode}: {len(pano_st2d3d_dataset)}')

        self.valid_iou_loader = DataLoader(
            pano_st2d3d_dataset,
            batch_size=self.cfg.model.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.model.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed(self.cfg.model.seed)
        )
        
    def set_valid_dataloader_lsun(self, lsun_dir, mode='val'):
        # assert mode == 'val', 'LSUN dataset only supports val mode'
        if mode == 'test':
            mode = 'val'
        logger.info(f"Setting IoU {mode} Dataloader")
        lsun_dataset = LSUNPreprocDataset(self.cfg.lsun.data_dir, mode, self.cfg.lsun.fix_shape)
        print(f'lsun_dataset_{mode}:', len(lsun_dataset))
        self.valid_iou_loader = DataLoader(
            lsun_dataset,
            batch_size=self.cfg.model.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.model.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed(self.cfg.model.seed)
        )

    def plot_pano(self, save_pred_boundary=False):
        self.net.eval()
        iterator_valid_iou = iter(self.valid_iou_loader)
        output_dir = os.path.join(self.cfg.output_dir, self.cfg.id_exp)
        dst_dir_est = output_dir + '/inference_img/' + 'panorama/'
        if save_pred_boundary:
            save_boundary_dir_est = output_dir + '/inference_img/' + 'panorama_pred_boundary/'
            pathlib.Path(save_boundary_dir_est).mkdir(parents=True, exist_ok=True)
        pathlib.Path(dst_dir_est).mkdir(parents=True, exist_ok=True)
        i = 0

        for _ in trange(len(iterator_valid_iou), desc="plot image epoch %d" % self.current_epoch):
            sample = next(iterator_valid_iou)
            x = sample['img']
            img_name = sample['img_name']
            y_bon_ref = sample['boundary']
            gt_corner = sample['corner']
            gt_type = sample['gt_type']
            gt_type = gt_type.int()

            with torch.no_grad():
                y_bon_est, _ = self.net(x.to(self.device))
                for image, gt, est, corner, name in zip(x, y_bon_ref.cpu().numpy(), y_bon_est.cpu().numpy(), gt_corner.cpu().numpy(), img_name):
                    img = image.detach().cpu().numpy().transpose([1, 2, 0])
                    img = (img.copy()*255).astype(np.uint8)
                    gt_pixel = ((gt/np.pi + 0.5)*img.shape[0]).round()
                    est_pixel = ((est/np.pi + 0.5)*img.shape[0]).round()
                    v_x = np.linspace(0, img.shape[1] - 1, img.shape[1]).astype(int)

                    # plot gt boundary
                    gt_pixel_ceiling = np.vstack((v_x, gt_pixel[0])).transpose()
                    plotXY(img, gt_pixel_ceiling, color=(255,0,0))
                    gt_pixel_floor = np.vstack((v_x, gt_pixel[1])).transpose()
                    plotXY(img, gt_pixel_floor, color=(255,0,0))

                    est_pixel_ceiling = np.vstack((v_x, est_pixel[0])).transpose()
                    plotXY(img, est_pixel_ceiling, color=(0,255,255))
                    est_pixel_floor = np.vstack((v_x, est_pixel[1])).transpose()
                    plotXY(img, est_pixel_floor, color=(0,255,255))
                    imwrite(dst_dir_est+f'est_{i}_{name}.jpg',(img).astype(np.uint8))

                    # save predicted boundary and gt corner
                    if save_pred_boundary:
                        corner = corner.T
                        # only save non-zero corner
                        corner = corner[corner[:, 0] != 0]
                        pred_json = {
                            'img_name': name,
                            'corner': corner.tolist(),
                            'boundary': est.tolist(),
                        }
                        
                        with open(save_boundary_dir_est + f'est_{i}.json', 'w') as f:
                            json.dump(pred_json, f, indent=4)
                    i+=1

    def plot_pp(self):
        self.net.eval()
        iterator_valid_iou = iter(self.valid_iou_loader)
        output_dir = os.path.join(self.cfg.output_dir, self.cfg.id_exp)
        dst_dir_est = output_dir + '/inference_img/' + 'perspective/'
        if self.cfg.mode == 'test':
           self.cfg.mode = 'val'
        img_dir = self.cfg.lsun.data_dir + f'/{self.cfg.mode}/img_aligned'
        pathlib.Path(dst_dir_est).mkdir(parents=True, exist_ok=True)
        shape=(512,512)
        i = 0

        for _ in trange(len(iterator_valid_iou), desc="plot image epoch %d" % self.current_epoch):
            sample = next(iterator_valid_iou)
            x = sample['img']
            y_bon_ref = sample['boundary']
            eval_range = sample['eval_range']
            gt_type = sample['gt_type']
            u_range = sample['u_range']
            data_name = sample['img_name']
            
            u_range = u_range.int()
            eval_range = eval_range.int()
            gt_type = gt_type.int()
            v_shift_pixel = sample['v_shift'].int() 
            with torch.no_grad():
                y_bon_est, _ = self.net(x.to(self.device))
                for img, gt, est, gt_range, img_range, img_type, v_shift_pixel, img_name in zip(x.cpu().numpy() , y_bon_ref.cpu().numpy(), y_bon_est.cpu().numpy(), eval_range.cpu().numpy(),
                                                                  u_range.cpu().numpy(), gt_type.cpu().numpy(), v_shift_pixel.cpu().numpy(), data_name):
                    
                    pred_range, est = find_predict_range(img, gt, est, img_range, img_type-1)
                    gt_b_map, pred_b_map = gen_pp_boundary_map(gt, est, gt_range, pred_range, gt_v_shifting=v_shift_pixel, shape=shape)
                    gt_boundary = pp_map2boundary(gt_b_map, shape=shape)
                    pred_boundary = pp_map2boundary(pred_b_map, shape=shape)
                    # 0: wall, 1: ceiling, 2: floor
                    img = read_image(img_dir + '/' + img_name + '.png', shape)
                    img = img*255
                    v_x = np.linspace(0, img.shape[1] - 1, img.shape[1]).astype(int)
                    gt_pixel_ceiling = np.vstack((v_x, gt_boundary[0])).transpose()
                    plotXY(img, gt_pixel_ceiling, color=(255,0,0))
                    gt_pixel_floor = np.vstack((v_x, gt_boundary[1])).transpose()
                    plotXY(img, gt_pixel_floor, color=(255,0,0))

                    est_pixel_ceiling = np.vstack((v_x, pred_boundary[0])).transpose()
                    plotXY(img, est_pixel_ceiling, color=(0,255,255))
                    est_pixel_floor = np.vstack((v_x, pred_boundary[1])).transpose()
                    plotXY(img, est_pixel_floor, color=(0,255,255))
                    
                    imwrite(dst_dir_est+f"est_{i}_{img_name}.jpg",(img).astype(np.uint8))
                    i+=1
        