import argparse
import hydra
import torch
import numpy as np
import random
import sys
import pdb

from loguru import logger
from model.wrapper_ulayout import WrapperuLayout

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@hydra.main(config_path="config", config_name="main_config", version_base="1.3")
def main(cfg):
    
    cfg.id_exp = "ulayout" + "_" + cfg.pano_dataset + "_" + cfg.pp_dataset + "_" + cfg.mode
    fix_seed(cfg.model.seed)
    model = WrapperuLayout(cfg)

    if cfg.mode == "train":
        # the training dataset and validation dataset are already included panorama dataset and perspective dataset.
        if cfg.pano_dataset == "mp3d":
            model.prepare_for_training_multi_dataset_mp3d(cfg.mp3d.data_dir)
            model.set_multi_valid_dataloader_mp3d(cfg.mp3d.data_dir, "val")
        elif cfg.pano_dataset == "pano": 
            # panocontext dataset (divide in train, val , test) + whole stanford2d3d dataset (all in train)
            model.prepare_for_training_multi_dataset_panost2d3d(cfg.pano.data_dir, subset="pano")
            model.set_multi_valid_dataloader_panost2d3d(cfg.pano.data_dir, subset="pano", mode="val")
        elif cfg.pano_dataset == "st2d3d":
            # stanford2d3d dataset (divide in train, val , test) + panocontext dataset (all in train)
            model.prepare_for_training_multi_dataset_panost2d3d(cfg.pano.data_dir, subset="st2d3d")
            model.set_multi_valid_dataloader_panost2d3d(cfg.pano.data_dir, subset="st2d3d", mode="val")
        else:
            raise ValueError("Invalid dataset for panorama. Choose from ['mp3d', 'pano', 'st2d3d']")
        model.valid_iou_loop()
        model.save_current_scores()
        while model.is_training:
            model.train_loop()
            model.valid_iou_loop()
            model.save_current_scores()

    elif cfg.mode == "val" or cfg.mode == "test":
        model.prepare_for_validation_multi_dataset()
        if cfg.pano_dataset == "mp3d":
            model.set_multi_valid_dataloader_mp3d(cfg.mp3d.data_dir, cfg.mode)
        elif cfg.pano_dataset == "pano":
            model.set_multi_valid_dataloader_panost2d3d(cfg.pano.data_dir, subset="pano", mode=cfg.mode)
        elif cfg.pano_dataset == "st2d3d":
            model.set_multi_valid_dataloader_panost2d3d(cfg.pano.data_dir, subset="st2d3d", mode=cfg.mode)
        else:
            raise ValueError("Invalid dataset for panorama. Choose from ['mp3d', 'pano', 'st2d3d']")
        model.valid_iou_loop()
    else:
        raise ValueError("Invalid mode. Choose from ['train', 'val', 'test']")
    
    if cfg.vis:
        # for panorama dataset, we visualize the panorama and perspective images in panorama format.
        if cfg.pano_dataset == "mp3d":
            model.set_valid_dataloader_mp3d(cfg.mp3d.data_dir, mode=cfg.mode)
        elif cfg.pano_dataset == "pano":
            model.set_valid_dataloader_panost2d3d(cfg.pano.data_dir, subset="pano", mode=cfg.mode)
        elif cfg.pano_dataset == "st2d3d":
            model.set_valid_dataloader_panost2d3d(cfg.pano.data_dir, subset="st2d3d", mode=cfg.mode)
        model.plot_pano(cfg.save_pred) # 3D layout only provide for panorama dataset.
        # besides, we also visualize the perspective images in perspective format.
        model.set_valid_dataloader_lsun(cfg.lsun.data_dir, mode=cfg.mode)
        model.plot_pp()

if __name__ == "__main__":
    # Set the logger level to DEBUG
    logger.remove()
    logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD}</green> | <cyan>{function}</cyan>:<magenta>{line}</magenta> | <white>{message}</white>",
    colorize=True
    )
    logger.info("Starting the script...")
    main()  # Pass the args to main function
    
    