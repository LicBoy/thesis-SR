"""
Dataest source:  http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
Instrustion:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript srgan.py
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *
from metrics import *
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch

def main():
    os.makedirs("images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=128, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=128, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available()

    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize generator and discriminator
    generator = GeneratorResNet()
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
    feature_extractor = FeatureExtractor()

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load(f"saved_models/generator_{opt.epoch}.pth"))
        discriminator.load_state_dict(torch.load(f"saved_models/discriminator_{opt.epoch}.pth"))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    dataloader = DataLoader(
        ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # ----------
    #  Training
    # ----------

    import matplotlib.pyplot as plt
    loss_G_history = []
    loss_D_history = []


    generator.eval()
    discriminator.eval()
    feature_extractor.eval()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, imgs in enumerate(dataloader):
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            sys.stdout.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
            )

            loss_G_history.append(loss_G.item())
            loss_D_history.append(loss_D.item())

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                # print(batches_done)
                # save_image(imgs_lr[0], f'comp/LR_{batches_done}.png', normalize=True)
                # save_image(gen_hr[0], f'comp/SR_{batches_done}.png', normalize=True)
                # Save image grid with upsampled inputs and SRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr[:5], scale_factor=4)
                gen_hr = make_grid(gen_hr[:5], nrow=1, normalize=True)
                imgs_lr = make_grid(imgs_lr[:5], nrow=1, normalize=True)
                img_grid = torch.cat((imgs_lr[:5], gen_hr[:5]), -1)
                save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

            if (batches_done + 1) % 300 == 0:
                plt.plot(loss_G_history, label='G_loss')
                plt.plot(loss_D_history, label='D_loss')
                plt.xlabel('Number of batches');
                plt.ylabel('Loss');
                plt.title('G vs D loss');
                plt.legend()
                plt.savefig(f'loss_plots/loss_plot_{batches_done}.png')
                plt.clf()


        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
            torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)


def calculate_psnr_ssim(reference_img_path, comparing_img_path):
    ref_img = np.array(Image.open(reference_img_path))
    comp_img = np.array(Image.open(comparing_img_path))
    psnr_score = calculate_psnr(ref_img, comp_img)
    ssim_score = calculate_ssim(ref_img, comp_img)
    return (psnr_score, ssim_score)


def compare_all_methods(sr_path, bicubic_path, gt_path):
    sr_gt_comp = calculate_psnr_ssim(sr_path, gt_path)
    bicubic_gt_comp = calculate_psnr_ssim(bicubic_path, gt_path)
    print(f'SR and GT: PSNR = {sr_gt_comp[0]} | SSIM = {sr_gt_comp[1]}\nBicubic and GT: PSNR = {bicubic_gt_comp[0]} | SSIM = {bicubic_gt_comp[1]}')



if __name__ == '__main__':
    main()