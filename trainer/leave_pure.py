import sys, os
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model_pure as model  # Import the model without transformers
import importlib
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
import copy
import yaml
import ctools
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
import argparse

import torch.utils.checkpoint as checkpoint  # Import checkpoint module
from efficientnet_pytorch import EfficientNet  # Import EfficientNet
from torch.utils.data import DataLoader  # Import DataLoader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(config):
    # ===============================> Setup <================================

    dataloader = importlib.import_module("reader." + config.train.reader)
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.train.device}")
        cudnn.benchmark = True
    else:
        exit("CUDA is not available.")
        # device = torch.device("cpu")
        # print("CUDA is not available. Using CPU instead.")

    data = config.train.data
    save = config.train.save
    params = config.train.params

    print("===> Read data <===")
    data, folder = ctools.readfolder(
                        data, 
                        [config.train.person], 
                        reverse=True
                    )

    savename = folder[config.train.person] 

    # Load the dataset
    dataset = dataloader.loader(
                    data, 
                    params.batch_size, 
                    shuffle=True, 
                    num_workers=6
                ).dataset

    # Split the dataset into training and validation sets
    val_split = 0.2
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=6)

    print("===> Model building <===")
    net = model.Model(config=edict(train=config.train, test=None), backbone=config.train.backbone)  # Pass config and backbone to Model
    net.train(); net.to(device)

    print(f"Using backbone: {config.train.backbone}")  # Print the backbone information

    # Check how many parameters are being updated
    total_params = count_parameters(net)
    print(f"Total parameters: {total_params}")
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    # Pretrain
    # pretrain = config.train.pretrain
    # if pretrain.enable:
    #     if hasattr(pretrain, 'path'):
    #         # Load EfficientNet-B0 weights
    #         if config.train.backbone == 'efficientnet-b0':
    #             efficientnet_checkpoint = EfficientNet.from_pretrained('efficientnet-b0').state_dict()
    #             net.base_model.load_state_dict(efficientnet_checkpoint, strict=False)
    #         else:
    #             net.load_state_dict(torch.load(pretrain.path, map_location=device), strict=False)
    #     else:
    #         print("Pretrain path is not specified in the configuration.")
    #         exit()

    print("===> optimizer building <===")
    optimizer = optim.Adam(
                    net.parameters(),
                    lr=params.lr,  # Use the learning rate from the config
                    betas=(0.9,0.999)
                )
  
    scheduler = optim.lr_scheduler.StepLR( 
                    optimizer, 
                    step_size=params.decay_step, 
                    gamma=params.decay
                )

    if params.warmup:
        scheduler = GradualWarmupScheduler( 
                        optimizer, 
                        multiplier=1, 
                        total_epoch=params.warmup, 
                        after_scheduler=scheduler
                    )

    # Print the initial learning rate
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")

    savepath = os.path.join(save.metapath, save.folder, config.train.backbone, f"checkpoint/{savename}")

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # =======================================> Training < ==========================
    print("===> Training <===")
    length = len(train_loader); total = length * params.epoch
    timer = ctools.TimeCounter(total)

    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(config) + '\n')

        for epoch in range(1, params.epoch+1):
            net.train()
            for i, (data, anno) in enumerate(train_loader):

                # ------------------forward--------------------
                data["face"] = data["face"].to(device)

                anno = anno.to(device)

                loss = net.loss(data, anno)

                # -----------------backward--------------------
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                rest = timer.step()/3600

                # -----------------loger----------------------
                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch}]: " +\
                          f"[{i}/{length}] " +\
                          f"loss:{loss}" +\
                          f"lr:{ctools.GetLR(optimizer)} "+\
                          f"rest time:{rest:.2f}h"

                    print(log); outfile.write(log + "\n")
                    sys.stdout.flush(); outfile.flush()

            scheduler.step()

            # Validation phase
            print("===> Validation <===")
            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, anno in val_loader:
                    data["face"] = data["face"].to(device)
                    anno = anno.to(device)
                    loss = net.loss(data, anno)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            log = f"Validation loss: {val_loss}"
            print(log); outfile.write(log + "\n")

            # Save the model at specified intervals
            if (epoch + 1) % config.train.save.step == 0:
                save_path = f"{save.metapath}/{save.folder}/{config.train.backbone}/{save.model_name}_epoch{epoch + 1}.pth"
                torch.save(net.state_dict(), save_path)
                print(f"Model saved to {save_path}")

        # Save the final model
        final_save_path = f"{save.metapath}/{save.folder}/{config.train.backbone}/{save.model_name}_final.pth"
        torch.save(net.state_dict(), final_save_path)
        print(f"Final model saved to {final_save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--train', type=str,
                        help='The source config for training.')

    parser.add_argument('-p', '--person', type=int,
                        help='The tested person.')

    args = parser.parse_args()

    config = edict(yaml.load(open(args.train), Loader=yaml.FullLoader))

    config.train.person = args.person
    
    print("=====================>> (Begin) Training params << =======================")

    print(ctools.DictDumps(config))

    print("=====================>> (End) Traning params << =======================")

    main(config)