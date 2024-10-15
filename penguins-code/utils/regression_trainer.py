from dataset.dataset import Crowd
from torch.utils.data import DataLoader
import torch
import logging
from utils.helper import SaveHandler, AverageMeter
from utils.trainer import Trainer
from model.vgg import vgg19
import numpy as np
import os
import time
from loss.ssim_loss import cal_avg_ms_ssim
import random


def cross_entropy_loss(gt_attn_map, att_map, k):
    loss = (gt_attn_map * torch.log(att_map + 1e-10) + k *
            (1 - gt_attn_map) * torch.log(1 - att_map + 1e-10)) * -1
    cross_entropy_loss = torch.mean(loss)
    return cross_entropy_loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Reg_Trainer(Trainer):
    def setup(self):
        args = self.args
        setup_seed(args.seed)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_device(0)
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            logging.info('Using {} gpus'.format(self.device_count))
        else:
            raise Exception('GPU is not available')

        self.d_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  crop_size=args.crop_size,
                                  d_ratio=self.d_ratio,
                                  method=x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          batch_size=(args.batch_size if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=(args.num_workers * self.device_count),
                                          pin_memory=(True if x == 'train' else False)) for x in ['train', 'val']}
        
        self.test_data = Crowd(os.path.join(args.data_dir, 'test'),
                                crop_size=args.crop_size,
                                d_ratio=self.d_ratio,
                                method='val')
        self.test_loader = DataLoader(self.test_data, batch_size=1, shuffle=False, pin_memory=False, num_workers=(args.num_workers * self.device_count))

        self.model = vgg19()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_list = SaveHandler(num=args.max_num)


    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.epochs):
            logging.info('-' * 40 + "Epoch:{}/{}".format(epoch, args.epochs - 1) + '-' * 40)
            self.epoch = epoch
            self.train_epoch()
            if epoch >= args.start_val and epoch % args.val_epoch == 0:
                self.val_epoch()
    
    def test_epoch(self):
        epoch_start = time.time()
        self.model.eval()
        epoch_res = []
        for inputs, gt_counts, name in self.test_loader:
            with torch.no_grad():
                inputs = inputs.to(self.device)
                assert inputs.shape[0] == 1
                den_map, b_map = self.model(inputs)
                res = gt_counts[0].item() - torch.sum(den_map * (b_map >= 0.5)).item()
                
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        logging.info('Epoch {} Test, MAE: {:.2f}, MSE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mae, mse, (time.time() - epoch_start)))

    def train_epoch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        epoch_ssim = AverageMeter()
        epoch_cls = AverageMeter()
        self.model.train()

        for step, (img, den_map, b_map) in enumerate(
                self.dataloaders['train']):
            inputs = img.to(self.device)
            gt_den_map = den_map.to(self.device)
            gt_bg_map = b_map.to(self.device)

            with torch.set_grad_enabled(True):
                N = inputs.shape[0]
                den_map, b_map = self.model(inputs)
                ssim_loss = cal_avg_ms_ssim(den_map * gt_bg_map, gt_den_map * gt_bg_map, level = 3)
                cls_loss = cross_entropy_loss(gt_bg_map, b_map, 0.5)

                loss = ssim_loss + cls_loss * 0.1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                gt_counts = torch.sum(gt_den_map.view(inputs.shape[0], -1), dim=1).detach().cpu().numpy()
                pre_count = torch.sum((den_map * (b_map >= 0.5)).view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gt_counts
                epoch_loss.update(loss.item(), N)
                epoch_mae.update(np.mean(np.abs(res)), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_cls.update(cls_loss.item(), N)
                epoch_ssim.update(ssim_loss.item(), N)

        logging.info('Epoch {} Train, Loss: {:.4f}, SSIM: {:.4f} Cls: {:.4f} MSE: {:.2f}, MAE: {:.2f}, Cost: {:.1f} sec'
                     .format(self.epoch, epoch_loss.getAvg(), epoch_ssim.getAvg(), epoch_cls.getAvg(), np.sqrt(epoch_mse.getAvg()), epoch_mae.getAvg(),
                             (time.time() - epoch_start)))

        if self.epoch % 5 == 0:
            model_state_dict = self.model.state_dict()
            save_path = os.path.join(self.save_dir, "{}_ckpt.tar".format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dict,
            }, save_path)
            self.save_list.append(save_path)

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()
        epoch_res = []
        for inputs, gt_counts, name in self.dataloaders['val']:
            with torch.no_grad():
                inputs = inputs.to(self.device)
                assert inputs.shape[0] == 1
                den_map, b_map = self.model(inputs)
                res = gt_counts[0].item() - torch.sum(den_map * (b_map >= 0.5)).item()
                
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        logging.info('Epoch {} Val, MAE: {:.2f}, MSE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mae, mse, (time.time() - epoch_start)))

        model_state_dict = self.model.state_dict()

        if (mae + mse) < self.best_mae + self.best_mse:
            self.best_mae = mae
            self.best_mse = mse
            torch.save(model_state_dict, os.path.join(self.save_dir, 'best_model.pth'))
            logging.info("Save best model: MAE: {:.2f} MSE:{:.2f} model epoch {}".format(mae, mse, self.epoch))
            self.test_epoch()

        print("Current best: MAE: {:.2f} MSE:{:.2f}".format(self.best_mae, self.best_mse))

