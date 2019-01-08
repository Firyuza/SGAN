from datetime import datetime
from easydict import EasyDict

cfg = EasyDict()

cfg.VANILLA = 'VANILLA'
cfg.WGAN = 'WGAN'

cfg.dataset = EasyDict()
cfg.dataset.dataset_name = 'celebA' # 'mnist'
cfg.dataset.data_dir = './data' if cfg.dataset.dataset_name == 'mnist' else '/home/firiuza/MachineLearning/celebA'
cfg.dataset.seed = 547

cfg.train = EasyDict()
cfg.train.num_epochs = 1000
cfg.train.batch_size = 128

cfg.train.valid_part = 0.2

cfg.train.learning_rate = 2e-4
cfg.train.beta1 = 0.5
cfg.train.weight_decay = 0.0005

cfg.train.z_dim = 100
cfg.train.nrof_classes = 10

cfg.train.loss_type = 'VANILLA' #'WGAN'
cfg.train.use_GP = False
cfg.train.gp_weight = 0.01

if cfg.dataset.dataset_name == 'mnist':
    cfg.train.output_height = 28
    cfg.train.output_width = 28
    cfg.train.output_channels = 1
elif cfg.dataset.dataset_name == 'celebA':
    cfg.train.input_height = 108
    cfg.train.input_width = 108
    cfg.train.output_height = 64
    cfg.train.output_width = 64
    cfg.train.output_channels = 3

cfg.train.generator_fc_dim = 1024
cfg.train.generator_feat_dim = 64

cfg.train.discriminator_fc_dim = 1024
cfg.train.discriminator_feat_dim = 64

cfg.train.batcn_norm_momentum = 1e-5

cfg.train.N_pairs = 5

cfg.train.gpu_memory_fraction = 1.0

cfg.train.run_directory = './train_models/run_%s/' % datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
cfg.train.logs_base_dir = cfg.train.run_directory + 'logs/'
cfg.train.models_base_dir = cfg.train.run_directory + 'models/'

cfg.train.restore_model_path = ''

cfg.data_augmentation = EasyDict()
if cfg.dataset.dataset_name == 'mnist':
    cfg.data_augmentation.image_size = 28
    cfg.data_augmentation.channel_size = 1
elif cfg.dataset.dataset_name == 'celebA':
    cfg.data_augmentation.image_size = 64
    cfg.data_augmentation.channel_size = 3

cfg.train.leaky_relu_aplha = 0.2
cfg.train.label_smooth = 0 #0.17

cfg.validation = {}
cfg.validation.validation_dir = cfg.train.run_directory
cfg.validation.batch_size = 256
