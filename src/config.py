class GANConfig:

    def __init__(self) -> None:
        #generator
        self.latent_vector_size = 100
        self.generator_features_number = 128
        self.generator_features_multipliers = [8, 4, 2]
        self.generator_block_type = 'transpose'  #transpose or upsample
        #discriminator
        self.discriminator_input_layer_type = "conv"

        self.discriminator_block_type = 'basic' #basic or dropout
        self.discriminator_features_number = 64
        self.discriminator_features_multipliers = [1, 2, 4, 8]

        self.discriminator_init_args = []
        self.discriminator_last_layer_type = "conv"
        self.discriminator_last_layer_activation = "sigmoid"

        #optimiziers
        self.learning_rate = 2e-4
        self.beta1 = 0.5 
        #input
        self.number_of_channels = 3
        self.image_size = 64

class GANDropoutConfig:
    def __init__(self) -> None:
        self.latent_vector_size = 100

        #discriminator
        self.discriminator_input_type = "dropout"

        self.discriminator_block_type = "dropout"

        self.discriminator_features_number = 16
        self.discriminator_features_multipliers = [1,2,4,8]
        self.number_of_channels = 3

        self.last_layer_type = "linear"
        self.last_layer_activation = "sigmoid"
        
        self.image_size = 64

class TrainConfig:
    def __init__(self) -> None:
        #training
        self.number_of_epochs = 40
        self.batch_size = 10

        # model saving
        self.folder_name = "dcgan_basic_test"
        self.root_path   =  "../trained_models/dcgan/out/"

        self.train_from_scratch = True
        self.load_epoch = 0
        self.epochs_to_save = 10

class InferenceConfig:
    def __init__(self) -> None:
        self.batch_size = 8
        self.root_path  = "../trained_models/dcgan/out/"
        self.models     = [ "dcgan_dropout_test"]#, "dcgan_normal_test"]
        self.configs    = [ gan_dropout_cfg]#, gan_cfg]

        self.epoch      = [20,30, 40]
        self.rows       = 2

        self.threeshold = 0.2
        self.batch_gen_size = 256
        self.gt = False

DEVICE = "cuda"

gan_cfg = GANConfig()
gan_dropout_cfg = GANDropoutConfig()

#gan_dropout_cfg.generator_block_type = "upsample"
train_cfg = TrainConfig()
inference_cfg = InferenceConfig()