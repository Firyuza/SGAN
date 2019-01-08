import numpy as np
import torch
from torch import nn

from config_SGAN import cfg
from ops import *

class Discriminator(torch.nn.Module):
    def __init__(self,
                 dataset_type,
                 is_training=True):
        super(Discriminator, self).__init__()

        if dataset_type == 'mnist':
            in_channels_size = {'conv1': cfg.data_augmentation.channel_size + cfg.train.nrof_classes,
                                'conv2': cfg.data_augmentation.channel_size + 2 * cfg.train.nrof_classes,
                                'fc1': 3636,
                                'fc2': cfg.train.discriminator_fc_dim + cfg.train.nrof_classes}

            self.conv1 = nn.Sequential(
                default_conv2d(in_channels_size['conv1'], cfg.data_augmentation.channel_size + cfg.train.nrof_classes),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.conv2 = nn.Sequential(
                default_conv2d(in_channels_size['conv2'], cfg.train.discriminator_feat_dim + cfg.train.nrof_classes),
                nn.BatchNorm2d(cfg.train.discriminator_feat_dim + cfg.train.nrof_classes,
                               momentum=cfg.train.batcn_norm_momentum,
                               track_running_stats=is_training),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.fc1 = nn.Sequential(
                nn.Linear(in_channels_size['fc1'], cfg.train.discriminator_fc_dim),
                nn.BatchNorm1d(cfg.train.discriminator_fc_dim,
                               momentum=cfg.train.batcn_norm_momentum,
                               track_running_stats=is_training),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.fc2 = nn.Linear(in_channels_size['fc2'], 1)

            self.out = nn.Sigmoid()

        elif dataset_type == 'celebA':
            in_channels_size = {'conv1': cfg.data_augmentation.channel_size,
                                'conv2': cfg.train.discriminator_feat_dim,
                                'conv3': cfg.train.discriminator_feat_dim * 2,
                                'conv4': cfg.train.discriminator_feat_dim * 4,
                                'fc': cfg.train.discriminator_feat_dim * 8 * 4 * 4}

            self.conv1 = nn.Sequential(
                default_conv2d(in_channels_size['conv1'], cfg.train.discriminator_feat_dim),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.conv2 = nn.Sequential(
                default_conv2d(in_channels_size['conv2'], cfg.train.discriminator_feat_dim * 2),
                nn.BatchNorm2d(cfg.train.discriminator_feat_dim * 2,
                               momentum=cfg.train.batcn_norm_momentum,
                               track_running_stats=is_training),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.conv3 = nn.Sequential(
                default_conv2d(in_channels_size['conv3'], cfg.train.discriminator_feat_dim * 4),
                nn.BatchNorm2d(cfg.train.discriminator_feat_dim * 4,
                               momentum=cfg.train.batcn_norm_momentum,
                               track_running_stats=is_training),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.conv4 = nn.Sequential(
                default_conv2d(in_channels_size['conv4'], cfg.train.discriminator_feat_dim * 8),
                nn.BatchNorm2d(cfg.train.discriminator_feat_dim * 8,
                               momentum=cfg.train.batcn_norm_momentum,
                               track_running_stats=is_training),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.fc = nn.Linear(in_channels_size['fc'], 1)

            self.out = nn.Sigmoid()

        return

    def forward(self, dataset_type, input_x, y=None):
        if dataset_type == 'mnist':
            y_vec = torch.zeros((len(y), cfg.train.nrof_classes), dtype=torch.float32)
            if torch.cuda.is_available():
                y_vec = y_vec.cuda()
            for i, label in enumerate(y):
                y_vec[i][int(y[i])] = 1.0
            yb = y_vec.view([-1, cfg.train.nrof_classes, 1, 1])
            x = conv_cond_concat(input_x, yb)

            conv2d_h0 = self.conv1(x)
            h0 = conv_cond_concat(conv2d_h0, yb)

            conv2d_h1 = self.conv2(h0)
            h1 = conv2d_h1.contiguous().view([-1, conv2d_h1.shape[1] * conv2d_h1.shape[2] * conv2d_h1.shape[3]])
            h1 = torch.cat([h1, y_vec], 1)

            fc2 = self.fc1(h1)
            h2 = torch.cat([fc2, y_vec], 1)

            h3 = self.fc2(h2)

            out = self.out(h3)
        elif dataset_type == 'celebA':
            h0 = self.conv1(input_x)
            h1 = self.conv2(h0)
            h2 = self.conv3(h1)
            h3 = self.conv4(h2)
            h3 = h3.view(h3.shape[0], -1)
            h4 = self.fc(h3)

            out = self.out(h4)
        else:
            raise Exception('Unknown dataset')
        
        return out