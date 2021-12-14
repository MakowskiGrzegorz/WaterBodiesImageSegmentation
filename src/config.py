class GANConfig:

    def __init__(self) -> None:
        #generator
        self.latent_vector_size = 100
        self.generator_features_number = 128
        self.generator_features_multipliers = [8, 4, 2]

        #discriminator
        self.discriminator_features_number = 64
        self.discriminator_features_multipliers = [1, 2, 4, 8]

        #optimiziers
        self.learning_rate = 2e-4
        self.beta1 = 0.5
        #input
        self.number_of_channels = 3
        self.image_size = 64

class TrainConfig:
    def __init__(self) -> None:
        #training
        self.number_of_epochs = 30
        self.batch_size = 10



DEVICE = "cuda"

gan_cfg = GANConfig()
train_cfg = TrainConfig()