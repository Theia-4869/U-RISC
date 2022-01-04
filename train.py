import os
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import DataParallel
import cv2
import albumentations as A
from torch.cuda.amp import autocast, GradScaler

from utils import URISC, Options, F_score, LR_Scheduler, focal_loss, fscore_loss, near_edge_loss


class Trainer:
    def __init__(self, args):
        self.args = args
        self.amp = True

        if "simple" in str.lower(args.dataset):
            self.dataset = "simple"
        else:
            assert "complex" in str.lower(args.dataset)
            self.dataset = "complex"

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_transform = A.Compose([
            A.RandomResizedCrop(interpolation=cv2.INTER_NEAREST, width=args.crop_size, height=args.crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            A.RandomBrightnessContrast(p=0.5),
            A.MultiplicativeNoise(p=0.5),

            A.ElasticTransform(interpolation=cv2.INTER_NEAREST, p=0.5),
            A.GridDistortion(interpolation=cv2.INTER_NEAREST, p=0.5),
            A.OpticalDistortion(interpolation=cv2.INTER_NEAREST, p=0.5),
            A.CoarseDropout(max_holes=2, max_height=64, max_width=64, min_holes=1, min_height=32, min_width=32, p=0.5),
        ])

        trainset = URISC(dir="data/datasets", mode="train", transform=train_transform,
                         data_rank=self.dataset)
        valset = URISC(dir="data/datasets", mode="val", transform=None,
                         data_rank=self.dataset)
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      drop_last=False, pin_memory=True, num_workers=16)
        self.valloader = DataLoader(valset, batch_size=4, shuffle=False, drop_last=False)

        if args.model == "UNet":
            from models import UNet
            self.model = DataParallel(UNet(backbone=args.backbone, amp=self.amp)).cuda()
        elif args.model == "ResNetUNet":
            from models import ResNetUNet
            self.model = DataParallel(ResNetUNet(backbone=args.backbone, amp=self.amp)).cuda()
        elif args.model == "CASENet":
            from models import CASENet
            self.model = DataParallel(CASENet(backbone=args.backbone, amp=self.amp)).cuda()
        elif args.model == "DDS":
            from models import DDS
            self.model = DataParallel(DDS(backbone=args.backbone, amp=self.amp)).cuda()
        elif args.model == "DFF":
            from models import DFF
            self.model = DataParallel(DFF(backbone=args.backbone, amp=self.amp)).cuda()
        else:
            # TODO add more models
            pass

        params_list = [{"params": self.model.module.backbone.parameters(), "lr": args.lr}]
        for name, param in self.model.module.named_parameters():
            if "backbone" not in name:
                params_list.append({"params": param, "lr": args.lr * args.lr_times})
        self.optimizer = Adam(params_list)
        self.lr_scheduler = LR_Scheduler(base_lr=args.lr, epochs=args.epochs,
                                         iters_each_epoch=len(self.trainloader), lr_times=args.lr_times)
        self.grad_scaler = GradScaler(enabled=self.amp)
        self.iterations = 0
        self.previous_best = 0.0

    def training(self, epoch):
        self.model.train()
        total_loss = 0.0
        tbar = tqdm(self.trainloader)
        for i, (image, target) in enumerate(tbar):
            self.lr_scheduler(self.optimizer, self.iterations, epoch)
            self.iterations += 1
            image, target = image.cuda(), target.cuda()
            
            with autocast(enabled=self.amp):
                predicted_logits,  predicted = self.model(image)
                loss = 0.5 * fscore_loss(predicted, target) + \
                    0.5 * focal_loss(predicted_logits, target, alpha=self.args.alpha, power=2) + \
                    0.4 * near_edge_loss(predicted, target, kernel_size=self.args.kernel_size)
            
            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            # loss.backward()
            # self.optimizer.step()

            total_loss += loss.item()
            tbar.set_description("Train loss: %.4f" % (total_loss / (i + 1)))

    def validation(self):
        self.model.eval()
        # evaluate on training set
        if args.eval_train:
            trainset = URISC(dir="data/datasets", mode="train", transform=None, data_rank=self.dataset)
            trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
            self.eval(trainloader)

        # evaluate on validation set
        mean_f_score = self.eval(self.valloader)

        if mean_f_score > self.previous_best:
            if not os.path.isdir(os.path.join(args.output, self.dataset, "models", args.model)):
                os.makedirs(os.path.join(args.output, self.dataset, "models", args.model))
            if self.previous_best != 0:
                os.remove(os.path.join(args.output, self.dataset, "models", args.model, "best_%.5f.pth" % self.previous_best))
            self.previous_best = mean_f_score
            torch.save(self.model.state_dict(), os.path.join(args.output, self.dataset, "models", args.model, "best_%.5f.pth" % self.previous_best))

    def eval(self, dataloader):
        total_f_score, total_precison, total_recall = 0.0, 0.0, 0.0
        tbar = tqdm(dataloader, desc="\r")
        for t, (image, target) in enumerate(tbar):
            if self.dataset == "simple":
                image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    predicted = self.model.module.tta_eval(image)
                    predicted = (predicted > 0.5).float()
                    f_score, precision, recall = F_score(predicted, target)
                    total_f_score += f_score
                    total_precison += precision
                    total_recall += recall
                    tbar.set_description("F-score: %.3f, Precision: %.3f, Recall: %.3f" %
                                        (total_f_score / (t + 1), total_precison / (t + 1), total_recall / (t + 1)))
            elif self.dataset == "complex":
                img_shape = image.shape
                tar_shape = target.shape
                padding_image = torch.zeros((img_shape[0], img_shape[1], 10240, 10240))
                padding_image[:, :, :img_shape[2], :img_shape[3]] = image
                padding_predicted = torch.zeros((tar_shape[0], tar_shape[1], 10240, 10240))

                for i in range(10):
                    for j in range(10):
                        part_image = padding_image[:, :, 1024*i:1024*(i+1), 1024*j:1024*(j+1)].cuda()
                        with torch.no_grad():
                            part_predicted = self.model.module.tta_eval(part_image)
                            part_predicted = (part_predicted > 0.5).float()
                            padding_predicted[:, :, 1024*i:1024*(i+1), 1024*j:1024*(j+1)] = part_predicted.cpu()

                f_score, precision, recall = F_score(padding_predicted[:, :, :tar_shape[2], :tar_shape[3]], target)
                total_f_score += f_score
                total_precison += precision
                total_recall += recall
                tbar.set_description("F-score: %.3f, Precision: %.3f, Recall: %.3f" %
                                    (total_f_score / (t + 1), total_precison / (t + 1), total_recall / (t + 1)))
            else:
                raise NotImplementedError("Wrong dataset!")
        mean_f_score = total_f_score / len(dataloader)
        return mean_f_score


if __name__ == "__main__":
    args = Options().parse()
    trainer = Trainer(args)
    print("Total Epoches: %i" % (args.epochs))
    for epoch in range(args.epochs):
        print("\n=>Epoches %i, learning rate = %.4f, \t\t\t\t previous best = %.4f"
              % (epoch, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))
        trainer.training(epoch)
        trainer.validation()
