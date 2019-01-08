import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

def default_conv2d(in_channels, filters, kernel_size=4, strides=2):
    return nn.Conv2d(
        in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
        stride=strides, padding=1, bias=False)

def default_conv2d_transpose(in_channels, filters, kernel_size=4, strides=(2, 2)):
    return nn.ConvTranspose2d(
        in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
        stride=strides, padding=1, bias=False)

def concat(tensors, axis):
    return torch.cat(tensors, axis)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.shape
  y_shapes = y.shape
  ones = torch.ones([x_shapes[0], y_shapes[1], x_shapes[2], x_shapes[3]])
  if torch.cuda.is_available():
      ones = ones.cuda()
  return concat([
    x, y*ones], 1)

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


def gradient_penalty(discriminator, real_data, fake_data, gp_weight, labels, dataset_name):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    if torch.cuda.is_available():
        alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    if torch.cuda.is_available():
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(dataset_name, interpolated, labels)

    ones_as_prob_interpolated = torch.ones(prob_interpolated.size())
    if torch.cuda.is_available():
        ones_as_prob_interpolated = ones_as_prob_interpolated.cuda()

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=ones_as_prob_interpolated,
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1).mean().data[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean(), gradient_norm