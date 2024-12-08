import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse
import torch.backends.cudnn as cudnn  # Add this import
from efficientnet_pytorch import EfficientNet  # Add this import

def main(train, test):

    # ===============================> Setup <============================
    reader = importlib.import_module("reader." + test.reader)

    data = test.data
    load = test.load
    torch.cuda.set_device(test.device)
    cudnn.benchmark = True  # Add this line

    # ==============================> Read Data <======================== 
    data, folder = ctools.readfolder(data, [test.person])

    testname = folder[test.person] 

    dataset = reader.loader(data, 500, num_workers=4, shuffle=True)

    # modelpath = os.path.join(train.save.metapath, 
    #                         train.save.folder, train.backbone, f'trans6_final.pth')
    # modelpath = train.pretrain.path  # Use the pretrain path for the model
    modelpath = './save/mpii/efficientnet-b0/leave_80_epoch10.pth'
    

    print(f"Model path: {modelpath}")  # Add this line to print the model path
    logpath = os.path.join(train.save.metapath, 
                                train.save.folder, f'{test.savename}/{testname}')

    if not os.path.exists(logpath):
        os.makedirs(logpath)
    print(f"Log path: {logpath}")

    # =============================> Test <==============================

    print("Testing final model") 
    print(f"Using backbone: {train.backbone}")

    # ----------------------Load Model------------------------------
    net = model.Model(config=edict(train=train, test=test), backbone=train.backbone)  # Pass the backbone argument

    statedict = torch.load(
        modelpath,
        map_location={f"cuda:{train.device}":f"cuda:{test.device}"}
    )

    if not isinstance(statedict, dict):
        raise ValueError(f"Loaded state_dict is not a dictionary. Got type: {type(statedict)}")

    net.cuda(); net.load_state_dict(statedict, strict=False); net.eval()  # Load with strict=False

    length = len(dataset); accs = 0; count = 0

    # -----------------------Open log file--------------------------------
    logname = "checkpoint_model.log"
    
    outfile =  open(os.path.join(logpath, logname), 'w')
    outfile.write("name results gts\n")
    print(f"Log file created: {logname}")

    # -------------------------Testing---------------------------------
    with torch.no_grad():

        for j, (data, label) in enumerate(dataset):
            print(f"Processing batch {j+1}/{len(dataset)}")

            for key in data:
                if key != 'name': data[key] = data[key].cuda()

            names =  data["name"]

            gts = label
            gazes = net(data)

            for k, gaze in enumerate(gazes):

                gaze = gaze.cpu().detach().numpy()
                gt = gts.numpy()[k]

                count += 1
                accs += gtools.angular(
                            gtools.gazeto3d(gaze), 
                            gtools.gazeto3d(gt)
                        )
        
                name = [names[k]]
                gaze = [str(u) for u in gaze] 
                gt = [str(u) for u in gt] 
                log = name + [",".join(gaze)] + [",".join(gt)]
                outfile.write(" ".join(log) + "\n")

        loger = f"[final_model] Total Num: {count}, avg: {accs/count}"
        outfile.write(loger)
        print(loger)
    outfile.close()
    print(f"Log file closed: {logname}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--source', type=str,
                        help = 'config path about training')

    parser.add_argument('-t', '--target', type=str,
                        help = 'config path about test')

    parser.add_argument('-p', '--person', type=int,
                        help = 'the num of subject for test')

    args = parser.parse_args()

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))
    test_conf = test_conf.test

    test_conf.person = args.person

    print("=======================>(Begin) Config of training<======================")

    print(ctools.DictDumps(train_conf))

    print("=======================>(End) Config of training<======================")

    print("")

    print("=======================>(Begin) Config for test<======================")

    print(ctools.DictDumps(test_conf))

    print("=======================>(End) Config for test<======================")

    main(train_conf.train, test_conf)

