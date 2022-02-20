class GANConfig:

    def __init__(self) -> None:
        #generator
        self.latent_vector_size = 100
        self.generator_features_number = 128
        self.generator_features_multipliers = [8, 4, 2]
        self.generator_block_type = 'transpose'  #transpose or upsample
        #discriminator
        self.discriminator_features_number = 64
        self.discriminator_features_multipliers = [1, 2, 4, 8]
        self.discriminator_block_type = 'basic' #basic or dropout
        #optimiziers
        self.learning_rate = 2e-4
        self.beta1 = 0.5
        #input
        self.number_of_channels = 3
        self.image_size = 64

class TrainConfig:
    def __init__(self) -> None:
        #training
        self.number_of_epochs = 15
        self.batch_size = 10

        # model saving
        self.folder_name = "dcgan_ekstra_test"
        self.root_path   =  "../trained_models/dcgan/out/"

        self.train_from_scratch = False
        self.load_epoch = 15
        self.epochs_to_save = 5

class InferenceConfig:
    def __init__(self) -> None:
        self.batch_size = 8
        self.root_path  = "../trained_models/dcgan/out/"
        self.models     = ["dcgan_ekstra_test", "dcgan_refactor_test"]
        self.configs    = [gan_cfg, gan_cfg]

        self.epoch      = 30
        self.rows       = 2

        self.threeshold = 0.2
        self.batch_gen_size = 256
        self.gt = True

DEVICE = "cuda"

gan_cfg = GANConfig()
train_cfg = TrainConfig()
inference_cfg = InferenceConfig()