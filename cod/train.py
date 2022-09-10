import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os, argparse
from tqdm import tqdm
import sys
sys.path.append('.')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Generator
from data import get_loader
from utils import adjust_lr, AvgMeter
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--trainsize', type=int, default=480, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
opt = parser.parse_args()

# build models
generator = Generator(channel=opt.feat_channel)
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

image_root = 'D:\\Github\\Data\\COD10K-v3\\Test\\Image\\'
gt_root = 'D:\\Github\\Data\\COD10K-v3\\Test\\GT_Object\\'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1]  # multi-scale training


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce=None)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def visualize_prediction_init(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        # 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
        # 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad
        # 这样我们就会继续使用这个新的tensor进行计算，后面当我们进行反向传播时，到该调用detach()的tensor就会停止，不能再继续向前进行传播
        # 即下面的pred_edge_kk不能反向传播，而上面的pred_edge_kk可以反向传播
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_init.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


def visualize_gt(var_map):
    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)


if __name__ == '__main__':
    for epoch in range(1, (opt.epoch + 1)):
        # scheduler.step()
        generator.train()
        loss_record = AvgMeter()
        print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
        iterration = tqdm(train_loader)
        for i, pack in enumerate(iterration, start=1):
            for rate in size_rates:
                generator_optimizer.zero_grad()
                images, gts = pack
                images = Variable(images)
                gts = Variable(gts)
                images = images.cuda()
                gts = gts.cuda()
                # multi-scale training samples
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                pred_post_init = generator.forward(images)

                sal_loss = structure_loss(pred_post_init, gts)

                sal_loss.backward()
                generator_optimizer.step()

                visualize_prediction_init(torch.sigmoid(pred_post_init))
                visualize_gt(gts)

                if rate == 1:
                    loss_record.update(sal_loss.data, opt.batchsize)

            if i % 10 == 0 or i == total_step:
                status = ('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}'.
                          format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))
                iterration.set_description(desc=status)

        adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

        save_path = 'models/Resnet/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % opt.epoch == 0:
            torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
