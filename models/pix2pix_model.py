"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import torch.nn.functional as F
import util.util as util


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_images, _ = self.generate_fake(input_semantics, real_image)
            return fake_images[-1]  # 返回最高分辨率的fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            # 修正为所有判别器参数
            if isinstance(self.netD, torch.nn.ModuleList):
                D_params = []
                for D in self.netD:
                    D_params += list(D.parameters())
            else: 
                D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        # 多判别器
        if isinstance(self.netD, torch.nn.ModuleList):
            for idx, D in enumerate(self.netD):
                util.save_network(D, f'D{idx}', epoch, self.opt)
        else:
            util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                if isinstance(netD, torch.nn.ModuleList):
                    for idx, subD in enumerate(netD):
                        netD[idx] = util.load_network(subD, f"D{idx}", opt.which_epoch, opt)
                else:
                    netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        # 生成多尺度fake_images
        fake_images, KLD_loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)  # fake_images 列表

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        # 多分辨率real_images和input_semanitcs_list
        num_scales = len(fake_images)
        # 一次性过所有判别器
        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_images, real_image
        )

        # ──────────── GAN/Feature Matching Loss ──────────────
        # 1. GAN损失，按多判别器加和或平均
        # 多尺度判别器有不同数量的head
        G_losses['GAN'] = 0
        gan_loss_weights = [0.05, 0.15, 1.0, 1.0]
        cnt = 0
        for di, pred_d in enumerate(pred_fake):    # di: discriminator idx，pred_d: list of heads
            for hi, head in enumerate(pred_d):     # hi: head idx
                # head: list，每一层输出，最后一层是判别结果
                # head[-1] 是判别结果
                weight = gan_loss_weights[cnt] if cnt < len(gan_loss_weights) else 1.0
                G_losses['GAN'] += self.criterionGAN(head[-1], True, for_discriminator=False) * weight
                cnt += 1

        # 2. Feature Matching Loss
        if not self.opt.no_ganFeat_loss:
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            # for k in range(num_scales):
            num_D = len(pred_fake[-1])
            for i in range(num_D):
                num_intermediate_outputs = len(pred_fake[-1][i]) - 1
                for j in range(num_intermediate_outputs):
                    loss = self.criterionFeat(pred_fake[-1][i][j], pred_real[-1][i][j].detach())
                    GAN_Feat_loss += loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        # 3. VGG perceptual loss（只对最高分辨率，可选多尺度求平均）
        if not self.opt.no_vgg_loss:
            # 推荐只对最高分辨率
            G_losses['VGG'] = self.criterionVGG(fake_images[-1], real_image) * self.opt.lambda_vgg

        return G_losses, fake_images

    def compute_discriminator_loss(self, input_semantics, real_image):
        """
        Args:
            input_semantics: 输入的语义分割图
            real_image: 高分辨率真图片

        假设你有 self.discriminators, self.criterionGAN, self.downsample_func
        """
        D_losses = {}
        with torch.no_grad():
            fake_images, _ = self.generate_fake(input_semantics, real_image)   # fake_images: list, [low, mid, high]
            fake_images = [img.detach() for img in fake_images]
        
        pred_fake_list, pred_real_list = self.discriminate(input_semantics, fake_images, real_image)
        for i in range(3):
            D_losses[f"D{i}_Fake"] = self.criterionGAN(pred_fake_list[i], False, for_discriminator=True)
            D_losses[f"D{i}_Real"] = self.criterionGAN(pred_real_list[i], True, for_discriminator=True)

        return D_losses

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_images = self.netG(input_semantics, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_images, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_images, real_image):
        preds_fake = []
        preds_real = []
        num_scales = len(fake_images)

        # 按照分辨率，对real_image做多级降采样的准备
        real_images = [real_image]
        for _ in range(1, num_scales):
            real_images.insert(0, F.interpolate(real_images[0], scale_factor=0.5, mode='bilinear', align_corners=False)) 
        # 现在 real_images[0]是最小，real_images[1]中分，real_images[2]最大

        for i in range(len(self.netD)):
            fake = fake_images[i]
            real = real_images[i]
            # 语义图要下采样，保证分辨率对齐
            input_semantics_i = F.interpolate(input_semantics, size=fake.shape[2:], mode='nearest')
            
            fake_concat = torch.cat([input_semantics_i, fake], dim=1)
            real_concat = torch.cat([input_semantics_i, real], dim=1)
            fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
            output = self.netD[i](fake_and_real)
            pred_fake, pred_real = self.divide_pred(output)
            preds_fake.append(pred_fake)
            preds_real.append(pred_real)
        return preds_fake, preds_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
