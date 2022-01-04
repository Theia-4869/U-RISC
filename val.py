from models import UNet, ResNetUNet, CASENet, DDS, DFF
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from tqdm import tqdm
import os
import torch
import numpy as np
import cv2
from utils import URISC, Options


if __name__ == "__main__":
    args = Options().parse()

    if "simple" in str.lower(args.dataset):
        args.dataset = "simple"
        img_size = 1024
    else:
        assert "complex" in str.lower(args.dataset)
        args.dataset = "complex"
        img_size = 10240

    valset = URISC(dir="data/datasets", mode="val", transform=None, data_rank=args.dataset)
    valloader = DataLoader(valset, batch_size=1, shuffle=False)

    model_dir = os.path.join("outdir", args.dataset, "models")
    model_names = os.listdir(model_dir)
    
    outdir = os.path.join("outdir", args.dataset, "val")
    ensemble_mask = torch.zeros((len(valloader), 1, 1, img_size, img_size))

    for i in range(len(model_names)):
        model_name = model_names[i]
        if os.path.isfile(os.path.join(model_dir, model_name)):
            continue
        if model_name == "UNet":
            model = DataParallel(UNet(backbone=args.backbone)).cuda()
        elif model_name == "ResNetUNet":
            model = DataParallel(ResNetUNet(backbone=args.backbone)).cuda()
        elif model_name == "CASENet":
            model = DataParallel(CASENet(backbone=args.backbone)).cuda()
        elif model_name == "DDS":
            model = DataParallel(DDS(backbone=args.backbone)).cuda()
        elif model_name == "DFF":
            model = DataParallel(DFF(backbone=args.backbone)).cuda()
        else:
            print("Not a proper model name")
            exit(0)

        ckps = os.listdir(os.path.join(model_dir, model_name))
        ckp = ckps[0]

        print("Loading model:", model_name, ckp)
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name, ckp)), strict=False)
        model.eval()
    
        tbar = tqdm(valloader, desc="\r")
        for t, (image, filename) in enumerate(tbar):
            if args.dataset == "simple":
                image = image.cuda()
                with torch.no_grad():
                    predicted = model.module.tta_eval(image).cpu()
                    ensemble_mask[t] += predicted
                    predicted = (predicted > (0.5)).float()
                predicted = predicted * 255.0
                mask = predicted.squeeze().detach().numpy().astype(np.uint8)
            elif args.dataset == "complex":
                img_shape = image.shape
                padding_image = torch.zeros((img_shape[0], img_shape[1], 10240, 10240))
                padding_image[:, :, :img_shape[2], :img_shape[3]] = image
                padding_predicted = torch.zeros((img_shape[0], 1, 10240, 10240))
                for i in range(10):
                    for j in range(10):
                        part_image = padding_image[:, :, 1024*i:1024*(i+1), 1024*j:1024*(j+1)].cuda()
                        with torch.no_grad():
                            part_predicted = model.module.tta_eval(part_image).cpu()
                            part_predicted = (part_predicted > 0.5).float()
                            padding_predicted[:, :, 1024*i:1024*(i+1), 1024*j:1024*(j+1)] = part_predicted
                            ensemble_mask[t][:, :, 1024*i:1024*(i+1), 1024*j:1024*(j+1)] += part_predicted
                predicted = padding_predicted[:, :, :img_shape[2], :img_shape[3]]
                predicted = predicted * 255.0
                mask = predicted.squeeze().detach().numpy().astype(np.uint8)
            
            filename = filename[0][filename[0].rfind("/")+1:]
            if not os.path.isdir(os.path.join(outdir, model_name)):
                os.makedirs(os.path.join(outdir, model_name))
            cv2.imwrite(os.path.join(outdir, model_name, filename), mask)

    ensemble_mask /= len(model_names)
    tbar = tqdm(valloader, desc="\r")
    for t, (image, filename) in enumerate(tbar):
        with torch.no_grad():
            predicted = (ensemble_mask[t][:, :, :9958, :9959] > 0.34).float()
            predicted = predicted * 255.0
            mask = predicted.cpu().squeeze().detach().numpy().astype(np.uint8)
            filename = filename[0][filename[0].rfind("/")+1:]
            if not os.path.isdir(os.path.join(outdir, "ensemble")):
                os.makedirs(os.path.join(outdir, "ensemble"))
            cv2.imwrite(os.path.join(outdir, "ensemble", filename), mask)
