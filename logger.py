import os
import numpy as np
import errno
import scipy.misc
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch

class Logger:
    def __init__(self, model_name, data_name, logdir=None):
        self.model_name = model_name
        self.data_name = data_name
        self.log_dir = logdir
        self.log_dir_validation = self.log_dir  + 'validation/'
        self.log_dir_models = self.log_dir + 'models/'

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(logdir, 'logs'), comment=self.comment)

    def log(self, d_error, g_error, epoch, n_batch, num_batches, name):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/{}/D_error'.format(self.comment, name), d_error, step)
        self.writer.add_scalar(
            '{}/{}/G_error'.format(self.comment, name), g_error, step)

    def log_score(self, score, epoch, n_batch, num_batches, name, score_type):
        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/{}/{}'.format(self.comment, score_type, name), score, step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches, type_GAN='pairs', format='NCHW', normalize=True):
        '''
        input images are expected in format (NCHW)
        '''
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)

        if format == 'NHWC':
            images = images.transpose(1, 3)

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True)
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(
            images, nrow=nrows, normalize=True, scale_each=True)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step) # horizontal_grid

        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch, type_GAN)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, type_GAN, plot_horizontal=True):
        if self.log_dir is None:
            out_dir = './data/images/{}'.format(self.data_subdir)
        else:
            out_dir = os.path.join(self.log_dir_validation + str(epoch), str(n_batch) + '/' + type_GAN)
        Logger._make_dir(out_dir)

        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, type_GAN, 'hori')
        plt.close()

        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch, type_GAN)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, type_GAN, comment=''):
        if self.log_dir is None:
            out_dir = './data/images/{}'.format(self.data_subdir)
        else:
            out_dir = os.path.join(self.log_dir_validation + str(epoch), str(n_batch) + '/' + type_GAN)
        Logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
                                                         comment, epoch, n_batch))

    def log_images2(self, images, epoch, n_batch, type_GAN='pairs', comment=''):
        def inverse_transform(images):
            return (images + 1.) / 2.
        size = self._image_manifold_size(images.shape[0])
        image = np.squeeze(self._merge(images, size))

        if self.log_dir is None:
            out_dir = './data/images/{}'.format(self.data_subdir)
        else:
            out_dir = os.path.join(self.log_dir_validation + str(epoch), str(n_batch) + '/' + type_GAN)
        Logger._make_dir(out_dir)
        image_path = '{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
                                                         comment, epoch, n_batch)
        return scipy.misc.imsave(image_path, image)

    def _image_manifold_size(self, num_images):
        manifold_h = int(np.floor(np.sqrt(num_images)))
        manifold_w = int(np.ceil(np.sqrt(num_images)))
        assert manifold_h * manifold_w == num_images
        return manifold_h, manifold_w

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]
        if (images.shape[3] in (3, 4)):
            c = images.shape[3]
            img = np.zeros((h * size[0], w * size[1], c))
            for idx, image in enumerate(images):
                i = idx % size[1]
                j = idx // size[1]
                img[j * h:j * h + h, i * w:i * w + w, :] = image
            return img
        elif images.shape[3] == 1:
            img = np.zeros((h * size[0], w * size[1]))
            for idx, image in enumerate(images):
                i = idx % size[1]
                j = idx // size[1]
                img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
            return img
        else:
            raise ValueError('in merge(images,size) images parameter '
                             'must have dimensions: HxW or HxWx3 or HxWx4')

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch, num_epochs, n_batch, num_batches)
        )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, generator, discriminator, epoch, type_GAN):
        if self.log_dir is None:
            out_dir = './data/models/{}'.format(self.data_subdir)
        else:
            out_dir = os.path.join(self.log_dir_models + str(epoch), type_GAN)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
