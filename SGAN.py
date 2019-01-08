import numpy as np
import os
import shutil
import time
import multiprocessing

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam

from discriminator import Discriminator
from generator import Generator
from config_SGAN import cfg
from logger import Logger
from data_loader import get_train_valid_loader
from ops import gradient_penalty
from metrics import Score

class SGAN:
    def __init__(self):
        self.read_dataset()
        if not os.path.exists(cfg.train.run_directory):
            os.makedirs(cfg.train.run_directory)
        with open(cfg.train.run_directory + 'params.txt', 'w') as f:
            f.write(str(vars(cfg)))

        self.build_model()

        return

    def read_dataset(self):
        self.train_loader, self.valid_loader = get_train_valid_loader(data_dir=cfg.dataset.data_dir,
                                                                      dataset_type=cfg.dataset.dataset_name,
                                                                      train_batch_size=cfg.train.batch_size,
                                                                      valid_batch_size=cfg.validation.batch_size,
                                                                      augment=False if cfg.dataset.dataset_name == 'mnist' else True,
                                                                      random_seed=cfg.dataset.seed,
                                                                      valid_size=cfg.train.valid_part,
                                                                      shuffle=True,
                                                                      show_sample=False,
                                                                      num_workers=multiprocessing.cpu_count(),
                                                                      pin_memory=False)

        return


    def real_data_target(self, size):
        '''
        Tensor containing ones, with shape = size
        '''
        data = Variable(torch.ones(size, 1))
        if torch.cuda.is_available(): return data.cuda()
        return data

    def fake_data_target(self, size):
        '''
        Tensor containing zeros, with shape = size
        '''
        data = Variable(torch.zeros(size, 1))
        if torch.cuda.is_available(): return data.cuda()
        return data

    def train_discriminator(self, discriminator, optimizer, real_data, fake_data, labels):
        # Reset gradients
        optimizer.zero_grad()

        # 1. Train on Real Data
        D_real = discriminator(cfg.dataset.dataset_name, real_data, labels)
        # Calculate error and backpropagate
        D_loss_real = self.loss(D_real, self.real_data_target(real_data.size(0)))
        D_loss_real.backward()

        # 2. Train on Fake Data
        D_fake = discriminator(cfg.dataset.dataset_name, fake_data, labels)
        # Calculate error and backpropagate
        D_loss_fake = self.loss(D_fake, self.fake_data_target(fake_data.size(0)))
        D_loss_fake.backward()

        if cfg.train.loss_type == cfg.VANILLA:
            D_loss = D_loss_real + D_loss_fake
        elif cfg.train.loss_type == cfg.WGAN:
            D_loss = D_loss_fake - D_loss_real
            if cfg.train.use_GP:
                grad_penalty, gradient_norm = gradient_penalty(discriminator, real_data, fake_data, cfg.train.gp_weight,
                                                               labels, cfg.dataset.dataset_name)
                D_loss += grad_penalty

        # Update weights with gradients
        optimizer.step()

        return D_real, D_fake, D_loss, D_loss_real, D_loss_fake

    def train_generator(self, generator, discriminator, optimizer, z_noise, labels):
        # Reset gradients
        optimizer.zero_grad()

        # Sample noise and generate fake data
        G_fake_data = generator(cfg.dataset.dataset_name, z_noise, labels)
        D_fake = discriminator(cfg.dataset.dataset_name, G_fake_data, labels)
        # Calculate error and backpropagate
        G_loss = self.loss(D_fake, self.real_data_target(D_fake.size(0)))
        if cfg.train.loss_type == cfg.WGAN:
            G_loss = -1 * G_loss
        G_loss.backward()
        # Update weights with gradients
        optimizer.step()
        # Return error
        return G_fake_data, G_loss

    def build_model(self):
        if cfg.train.loss_type == cfg.VANILLA:
            self.loss = nn.BCELoss()
        elif cfg.train.loss_type == cfg.WGAN:
            self.loss = lambda logits, labels: torch.mean(logits)

        self.D_global = Discriminator(cfg.dataset.dataset_name)
        self.G_global = Generator(cfg.dataset.dataset_name)

        # Enable cuda if available
        if torch.cuda.is_available():
            self.D_global.cuda()
            self.G_global.cuda()

        # Optimizers
        self.D_global_optimizer = Adam(self.D_global.parameters(), lr=cfg.train.learning_rate, betas=(cfg.train.beta1, 0.999))
        self.G_global_optimizer = Adam(self.G_global.parameters(), lr=cfg.train.learning_rate, betas=(cfg.train.beta1, 0.999))

        self.D_pairs = []
        self.G_pairs = []
        self.D_pairs_optimizers = []
        self.G_pairs_optimizers = []

        self.D_msg_pairs = []
        self.D_msg_pairs_optimizers = []
        for id in range(1, cfg.train.N_pairs + 1):
            discriminator = Discriminator(cfg.dataset.dataset_name)
            generator = Generator(cfg.dataset.dataset_name)

            # Enable cuda if available
            if torch.cuda.is_available():
                generator.cuda()
                discriminator.cuda()

            self.D_pairs.append(discriminator)
            self.G_pairs.append(generator)
            
            # Optimizers
            D_optimizer = Adam(discriminator.parameters(), lr=cfg.train.learning_rate, betas=(cfg.train.beta1, 0.999))
            G_optimizer = Adam(generator.parameters(), lr=cfg.train.learning_rate, betas=(cfg.train.beta1, 0.999))

            self.D_pairs_optimizers.append(D_optimizer)
            self.G_pairs_optimizers.append(G_optimizer)

            # create msg Discriminator pair for G_global
            discriminator = Discriminator(cfg.dataset.dataset_name)

            # Enable cuda if available
            if torch.cuda.is_available():
                generator.cuda()
                discriminator.cuda()

            self.D_msg_pairs.append(discriminator)

            # Optimizers
            D_optimizer = Adam(discriminator.parameters(), lr=cfg.train.learning_rate, betas=(cfg.train.beta1, 0.999))

            self.D_msg_pairs_optimizers.append(D_optimizer)

        self.logger = Logger(model_name='DCGAN', data_name='MNIST', logdir=cfg.validation.validation_dir)


        return

    def run_validation(self, generator, discriminator, epoch, i, type_GAN):
        nrof_batches = len(self.valid_loader)
        for batch_idx, (valid_batch_images, valid_batch_labels) in enumerate(self.valid_loader):
            valid_batch_size = len(valid_batch_images)
            valid_batch_labels = valid_batch_labels.type(torch.float32)
            valid_batch_z = torch.from_numpy(np.random.uniform(-1, 1, [valid_batch_size, cfg.train.z_dim]).astype(np.float32))

            if torch.cuda.is_available():
                valid_batch_images = valid_batch_images.cuda()
                valid_batch_labels = valid_batch_labels.cuda()
                valid_batch_z = valid_batch_z.cuda()

            G_fake_data = generator(cfg.dataset.dataset_name, valid_batch_z, valid_batch_labels)
            D_fake = discriminator(cfg.dataset.dataset_name, G_fake_data, valid_batch_labels)
            G_loss = self.loss(D_fake, self.real_data_target(D_fake.size(0)))

            D_real = discriminator(cfg.dataset.dataset_name, valid_batch_images, valid_batch_labels)
            D_loss_real = self.loss(D_real, self.real_data_target(valid_batch_images.size(0)))
            D_fake = discriminator(cfg.dataset.dataset_name, G_fake_data, valid_batch_labels)
            D_loss_fake = self.loss(D_fake, self.fake_data_target(D_fake.size(0)))
            D_loss = D_loss_real + D_loss_fake

            if len(valid_batch_images) == cfg.validation.batch_size:
                inception_score, std = Score.inception_score(G_fake_data)
                self.logger.log_score(inception_score, epoch, batch_idx, nrof_batches, type_GAN, 'IS_validation')

            # self.logger.log_images(generated_images, valid_batch_size, epoch, val_i, nrof_valid_batches,
            #                        type_GAN='pairs', format='NHWC')
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (D_loss, G_loss))
            if batch_idx > 0 and batch_idx % 15 == 0:
                generated_images = G_fake_data.detach().cpu()
                generated_images = generated_images.permute([0, 2, 3, 1])
                self.logger.log_images2(generated_images, epoch, batch_idx, type_GAN=type_GAN)

            batch_idx += 1

            # self.logger.save_models(self.G_pairs[id], self.D_pairs[id], epoch, 'pairs')
        return

    def copy_network_parameters(self, src_network, dest_network):
        params_src = src_network.named_parameters()
        params_dest = dest_network.named_parameters()

        dict_dest_params = dict(params_dest)

        for name_src, param_src in params_src:
            if name_src in dict_dest_params:
                dict_dest_params[name_src].data.copy_(param_src.data)
        return

    def run_train(self):
        for epoch in range(cfg.train.num_epochs):
            for id in range(cfg.train.N_pairs):
                print('Train pairs')
                self.train_pairs_epoch(id, epoch)
                self.copy_network_parameters(self.D_pairs[id], self.D_msg_pairs[id])
                self.train_G_global_epoch(id, epoch)
                self.train_D_global_epoch(id, epoch)
                self.run_validation(self.G_global, self.D_global, epoch, None, 'global_pair')
                self.logger.save_models(self.G_global, self.D_global, epoch, 'global_pair')
        return

    def train_D_global_epoch(self, id, epoch):
        # torch.set_default_tensor_type('torch.DoubleTensor')
        nrof_batches = len(self.train_loader)
        train_time = 0
        for batch_idx, (batch_images, batch_labels) in enumerate(self.train_loader):
            start_time = time.time()
            batch_size = len(batch_images)
            batch_labels = batch_labels.type(torch.float32)
            batch_z = torch.from_numpy(np.random.uniform(-1, 1, [batch_size, cfg.train.z_dim]).astype(np.float32))

            # 1. Train Discriminator
            if torch.cuda.is_available():
                batch_images = batch_images.cuda()
                batch_labels = batch_labels.cuda()
                batch_z = batch_z.cuda()
            # Generate fake data
            G_fake_data = self.G_pairs[id](cfg.dataset.dataset_name, batch_z, batch_labels).detach()
            # Train D
            D_real, D_fake, D_loss, D_loss_real, D_loss_fake = self.train_discriminator(self.D_global, self.D_global_optimizer,
                                                                                        batch_images, G_fake_data, batch_labels)

            # 2. Train Generator
            G_fake_data, G_loss = self.train_generator(self.G_pairs[id], self.D_global, self.G_pairs_optimizers[id], batch_z, batch_labels)

            # 3. Train Discriminator twice
            # Generate fake data
            G_fake_data = self.G_pairs[id](cfg.dataset.dataset_name, batch_z, batch_labels).detach()
            # Train D
            D_real, D_fake, D_loss, D_loss_real, D_loss_fake = self.train_discriminator(self.D_global, self.D_global_optimizer,
                                                                                        batch_images, G_fake_data, batch_labels)

            # Log error
            self.logger.log(D_loss, G_loss, epoch, batch_idx, nrof_batches, 'D0-' + str(id + 1))

            if len(batch_images) == cfg.train.batch_size:
                inception_score, std = Score.inception_score(G_fake_data)
                self.logger.log_score(inception_score, epoch, batch_idx, nrof_batches, 'D0-' + str(id + 1), 'IS')

            duration = time.time() - start_time
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, cfg.train.num_epochs, batch_idx, nrof_batches,
                     time.time() - start_time, D_loss, G_loss))
            train_time += duration
            if batch_idx > 0 and batch_idx % 101 == 0:
                self.run_validation(self.G_pairs[id], self.D_global, epoch, batch_idx, 'D_global_pairs-' + str(id + 1))
            batch_idx += 1

        self.logger.save_models(self.G_pairs[id], self.D_global, epoch, 'D_global_pairs-' + str(id + 1))

        return

    def train_G_global_epoch(self, id, epoch):
        # torch.set_default_tensor_type('torch.DoubleTensor')
        nrof_batches = len(self.train_loader)
        train_time = 0
        for batch_idx, (batch_images, batch_labels) in enumerate(self.train_loader):
            start_time = time.time()
            batch_size = len(batch_images)
            batch_labels = batch_labels.type(torch.float32)
            batch_z = torch.from_numpy(np.random.uniform(-1, 1, [batch_size, cfg.train.z_dim]).astype(np.float32))

            # 1. Train Discriminator
            if torch.cuda.is_available():
                batch_images = batch_images.cuda()
                batch_labels = batch_labels.cuda()
                batch_z = batch_z.cuda()
            # Generate fake data
            G_fake_data = self.G_global(cfg.dataset.dataset_name, batch_z, batch_labels).detach()
            # Train D
            D_real, D_fake, D_loss, D_loss_real, D_loss_fake = self.train_discriminator(self.D_msg_pairs[id], self.D_msg_pairs_optimizers[id],
                                                                                        batch_images, G_fake_data, batch_labels)

            # 2. Train Generator
            G_fake_data, G_loss = self.train_generator(self.G_global, self.D_msg_pairs[id], self.G_global_optimizer, batch_z, batch_labels)

            # 3. Train Discriminator twice
            # Generate fake data
            G_fake_data = self.G_global(cfg.dataset.dataset_name, batch_z, batch_labels).detach()
            # Train D
            D_real, D_fake, D_loss, D_loss_real, D_loss_fake = self.train_discriminator(self.D_msg_pairs[id], self.D_msg_pairs_optimizers[id],
                                                                                        batch_images, G_fake_data, batch_labels)

            # Log error
            self.logger.log(D_loss, G_loss, epoch, batch_idx, nrof_batches, 'G0-' + str(id + 1))

            if len(batch_images) == cfg.train.batch_size:
                inception_score, std = Score.inception_score(G_fake_data)
                self.logger.log_score(inception_score, epoch, batch_idx, nrof_batches, 'G0-' + str(id + 1), 'IS')

            duration = time.time() - start_time
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, cfg.train.num_epochs, batch_idx, nrof_batches,
                     time.time() - start_time, D_loss, G_loss))
            train_time += duration
            if batch_idx > 0 and batch_idx % 101 == 0:
                self.run_validation(self.G_global, self.D_msg_pairs[id], epoch, batch_idx, 'G_global_pairs-' + str(id + 1))
            batch_idx += 1

        self.logger.save_models(self.G_global, self.D_msg_pairs[id], epoch, 'G_global_pairs-' + str(id + 1))

        return

    def train_pairs_epoch(self, id, epoch):
        nrof_batches = len(self.train_loader)
        train_time = 0
        for batch_idx, (batch_images, batch_labels) in enumerate(self.train_loader):
            start_time = time.time()
            batch_size = len(batch_images)
            batch_labels = batch_labels.type(torch.float32)
            batch_z = torch.from_numpy(np.random.uniform(-1, 1, [batch_size, cfg.train.z_dim]).astype(np.float32))

            # 1. Train Discriminator
            if torch.cuda.is_available():
                batch_images = batch_images.cuda()
                batch_labels = batch_labels.cuda()
                batch_z = batch_z.cuda()
            # Generate fake data
            G_fake_data = self.G_pairs[id](cfg.dataset.dataset_name, batch_z, batch_labels).detach()
            # Train D
            D_real, D_fake, D_loss, D_loss_real, D_loss_fake = self.train_discriminator(self.D_pairs[id],
                                                                                        self.D_pairs_optimizers[id],
                                                                                        batch_images, G_fake_data,
                                                                                        batch_labels)

            # 2. Train Generator
            G_fake_data, G_loss = self.train_generator(self.G_pairs[id], self.D_pairs[id], self.G_pairs_optimizers[id],
                                                       batch_z, batch_labels)

            # 3. Train Discriminator twice
            # Generate fake data
            G_fake_data = self.G_pairs[id](cfg.dataset.dataset_name, batch_z, batch_labels).detach()
            # Train D
            D_real, D_fake, D_loss, D_loss_real, D_loss_fake = self.train_discriminator(self.D_pairs[id],
                                                                                        self.D_pairs_optimizers[id],
                                                                                        batch_images, G_fake_data,
                                                                                        batch_labels)

            # Log error
            self.logger.log(D_loss, G_loss, epoch, batch_idx, nrof_batches, str(id + 1))

            if len(batch_images) == cfg.train.batch_size:
                inception_score, std = Score.inception_score(G_fake_data)
                self.logger.log_score(inception_score, epoch, batch_idx, nrof_batches, str(id + 1), 'IS')

            duration = time.time() - start_time
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, cfg.train.num_epochs, batch_idx, nrof_batches,
                     time.time() - start_time, D_loss, G_loss))
            train_time += duration
            if batch_idx > 0 and batch_idx % 101 == 0:
                self.run_validation(self.G_pairs[id], self.D_pairs[id], epoch, batch_idx, 'pairs-' + str(id + 1))

        self.logger.save_models(self.G_pairs[id], self.D_pairs[id], epoch, 'pairs-' + str(id + 1))

        return