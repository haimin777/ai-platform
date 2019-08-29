# generator idea from https://machinelearningmastery.com


#####################

# use MLflow

#####################
import mlflow.keras

mlflow.keras.autolog()

from keras.datasets.mnist import load_data

# load data
(trainX, trainy), (testX, testy) = load_data()


from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, LeakyReLU, Conv2DTranspose, Reshape
from keras.optimizers import Adam
import numpy as np

from numpy import ones
from numpy import zeros, vstack
from numpy.random import randn
from numpy.random import randint



#@click.command(help="Trains an GAN Keras model on MNIST dataset."
                  #  "The model and its metrics are logged with mlflow.")
#@click.option("--epochs", type=click.INT, default=10, help="Number of training epochs")

class mnist_GAN_Generator(object):

    def __init__(self, X_real):

        self.X = np.expand_dims(X_real, -1).astype('float32') / 255.0
        self.latent_dims = 100

    def define_discriminator(self, in_shape=(28, 28, 1)):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def define_generator(self):
        model = Sequential()
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        model.add(Dense(n_nodes, input_dim=self.latent_dims))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 128)))
        # upsample to 14x14
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 28x28
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
        return model

        # define the combined generator and discriminator model, for updating the generator

    def define_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(g_model)
        # add the discriminator
        model.add(d_model)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    # generate points in latent space as input for the generator
    def generate_latent_points(self, n_samples):
        # generate points in the latent space
        x_input = randn(self.latent_dims * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, self.latent_dims)
        return x_input

    def generate_real_samples(self, n_samples):
        # choose random instances
        ix = randint(0, self.X.shape[0], n_samples)
        # retrieve selected images
        X = self.X[ix]
        # generate 'real' class labels (1)
        y = ones((n_samples, 1))
        return X, y

    def generate_fake_samples(self, g_model, n_samples):
        # generate points in latent space
        x_input = self.generate_latent_points(n_samples)
        # predict outputs
        X = g_model.predict(x_input)
        # create 'fake' class labels (0)
        y = zeros((n_samples, 1))
        return X, y

    # train the generator and discriminator
    def train(self, g_model, d_model, gan_model, n_epochs=100, n_batch=256):
        bat_per_epo = int(self.X.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                X_real, y_real = self.generate_real_samples(half_batch)
                # generate 'fake' examples
                X_fake, y_fake = self.generate_fake_samples(g_model, half_batch)
                # create training set for the discriminator
                X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
                # update discriminator model weights
                d_loss, _ = d_model.train_on_batch(X, y)
                # prepare points in latent space as input for the generator
                X_gan = self.generate_latent_points(n_batch)
                # create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
                # summarize loss on this batch
                print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss, g_loss))

        return g_model

    def plot_results(self, g_model):
        X, _ = self.generate_fake_samples(g_model, n_samples=25)
        # plot the generated samples
        for i in range(n_samples):
            # define subplot
            plt.subplot(5, 5, 1 + i)
            # turn off axis labels
            plt.axis('off')
            # plot single image
            plt.imshow(X[i, :, :, 0], cmap='gray_r')
        # show the figure
        plt.show()


if __name__ == '__main__':


    gan_obj = mnist_GAN_Generator(trainX)

    g_model = gan_obj.define_generator()
    d_model = gan_obj.define_discriminator()
    gan_model = gan_obj.define_gan(g_model, d_model)

    #use generation model to generate synthetic MNIST digits

    generation_model = gan_obj.train(g_model, d_model, gan_model)
    gan_obj.plot_results(generation_model)


