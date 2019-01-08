import torch
from torch import nn
import torch.nn.functional as F

from config_SGAN import cfg
from ops import *

class Generator(torch.nn.Module):
    def __init__(self,
                 dataset_type,
                 is_training=True):
        super(Generator, self).__init__()

        if dataset_type == 'mnist':
            in_channels_size = {'dense1': cfg.train.z_dim + cfg.train.nrof_classes,
                                'dense2': cfg.train.generator_fc_dim + cfg.train.nrof_classes,
                                'deconv1': cfg.train.generator_feat_dim * 2 + cfg.train.nrof_classes,
                                'deconv2': cfg.train.generator_feat_dim * 2 + cfg.train.nrof_classes}

            self.s_h2, self.s_h4 = int(cfg.train.output_height / 2), int(cfg.train.output_height / 4)
            self.s_w2, self.s_w4 = int(cfg.train.output_width / 2), int(cfg.train.output_width / 4)

            self.dense1 = nn.Sequential(
                nn.Linear(in_channels_size['dense1'], cfg.train.generator_fc_dim),
                nn.BatchNorm1d(cfg.train.generator_fc_dim,
                               momentum=cfg.train.batcn_norm_momentum,
                               track_running_stats=is_training),
                nn.ReLU(inplace=True)
            )

            self.dense2 = nn.Sequential(
                nn.Linear(in_channels_size['dense2'], cfg.train.generator_feat_dim * 2 * self.s_h4 * self.s_w4),
                nn.BatchNorm1d(cfg.train.generator_feat_dim * 2 * self.s_h4 * self.s_w4,
                               momentum=cfg.train.batcn_norm_momentum,
                               track_running_stats=is_training),
                nn.ReLU(inplace=True)
            )

            self.deconv1 = nn.Sequential(
                default_conv2d_transpose(in_channels_size['deconv1'], cfg.train.generator_feat_dim * 2, 4),
                nn.BatchNorm2d(cfg.train.generator_feat_dim * 2,
                               momentum=cfg.train.batcn_norm_momentum,
                               track_running_stats=is_training),
                nn.ReLU(inplace=True)
            )

            self.deconv2 = nn.Sequential(
                default_conv2d_transpose(in_channels_size['deconv2'], cfg.train.output_channels, 4)
            )

            self.out = torch.nn.Sigmoid()
        elif dataset_type == 'celebA':
            self.s_h, self.s_w = cfg.train.output_height, cfg.train.output_width
            self.s_h2, self.s_w2 = conv_out_size_same(self.s_h, 2), conv_out_size_same(self.s_w, 2)
            self.s_h4, self.s_w4 = conv_out_size_same(self.s_h2, 2), conv_out_size_same(self.s_w2, 2)
            self.s_h8, self.s_w8 = conv_out_size_same(self.s_h4, 2), conv_out_size_same(self.s_w4, 2)
            self.s_h16, self.s_w16 = conv_out_size_same(self.s_h8, 2), conv_out_size_same(self.s_w8, 2)
            in_channels_size = {'dense1': cfg.train.z_dim,
                                'deconv1': cfg.train.generator_feat_dim * 8,
                                'deconv2': cfg.train.generator_feat_dim * 4,
                                'deconv3': cfg.train.generator_feat_dim * 2,
                                'deconv4': cfg.train.generator_feat_dim * 1}

            self.dense1 = nn.Linear(in_channels_size['dense1'], cfg.train.generator_feat_dim * 8 * self.s_h16 * self.s_w16)

            self.dense1_out = nn.Sequential(
                nn.BatchNorm2d(cfg.train.generator_feat_dim * 8,
                               momentum=cfg.train.batcn_norm_momentum,
                               track_running_stats=is_training),
                nn.ReLU(inplace=True)
            )

            self.deconv1 = nn.Sequential(
                default_conv2d_transpose(in_channels_size['deconv1'], cfg.train.generator_feat_dim * 4, 4),
                nn.BatchNorm2d(cfg.train.generator_feat_dim * 4,
                               momentum=cfg.train.batcn_norm_momentum,
                               track_running_stats=is_training),
                nn.ReLU(inplace=True)
            )

            self.deconv2 = nn.Sequential(
                default_conv2d_transpose(in_channels_size['deconv2'], cfg.train.generator_feat_dim * 2, 4),
                nn.BatchNorm2d(cfg.train.generator_feat_dim * 2,
                               momentum=cfg.train.batcn_norm_momentum,
                               track_running_stats=is_training),
                nn.ReLU(inplace=True)
            )

            self.deconv3 = nn.Sequential(
                default_conv2d_transpose(in_channels_size['deconv3'], cfg.train.generator_feat_dim * 1, 4),
                nn.BatchNorm2d(cfg.train.generator_feat_dim * 1,
                               momentum=cfg.train.batcn_norm_momentum,
                               track_running_stats=is_training),
                nn.ReLU(inplace=True)
            )

            self.deconv4 = default_conv2d_transpose(in_channels_size['deconv4'], cfg.train.output_channels, 4)

            self.out = nn.Tanh()
        else:
            raise Exception('Unknown dataset')

        return

    def forward(self, dataset_type, z, y=None):
        if dataset_type == 'mnist':
            y_vec = torch.zeros((len(y), cfg.train.nrof_classes), dtype=torch.float32)
            if torch.cuda.is_available():
                y_vec = y_vec.cuda()
            for i, label in enumerate(y):
                y_vec[i][int(y[i])] = 1.0
            yb = y_vec.view([-1, cfg.train.nrof_classes, 1, 1])
            z = concat([z, y_vec], 1)

            fc_h0 = self.dense1(z)
            h0 = concat([fc_h0, y_vec], 1)

            fc_h1 = self.dense2(h0)
            h1 = fc_h1.view([-1, cfg.train.generator_feat_dim * 2, self.s_h4, self.s_w4])
            h1 = conv_cond_concat(h1, yb)

            deconv1 = self.deconv1(h1)
            h2 = conv_cond_concat(deconv1, yb)

            deconv2 = self.deconv2(h2)

            out = self.out(deconv2)
        elif dataset_type == 'celebA':
            # project `z` and reshape
            fc1 = self.dense1(z)
            fc1 = fc1.view([-1, cfg.train.generator_feat_dim * 8, self.s_h16, self.s_w16])
            fc1 = self.dense1_out(fc1)

            deconv1 = self.deconv1(fc1)
            deconv2 = self.deconv2(deconv1)
            deconv3 = self.deconv3(deconv2)
            deconv4 = self.deconv4(deconv3)

            out = self.out(deconv4)
        else:
            raise Exception('Unknown dataset')

        return out