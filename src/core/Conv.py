from keras.layers import *
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras import losses
from .layer import stack_layers
from . import costs
from .util import get_scale




class ConvAE:

    def __init__(self,x,params):
        self.x = x
        a = tf.shape(self.x)[0]
        self.P = tf.eye(tf.shape(self.x)[0])
        h = x



        filters = params['filters']
        latent_dim = params['latent_dim']
        num_classes = params['n_clusters']
        self.Dy = tf.placeholder(tf.float32, [None, latent_dim], name='Dy')
        self.Dy1 = tf.placeholder(tf.float32, [None, num_classes], name='Dy1')
        for i in range(1):
            filters *= 2

            h = Conv2D(filters=filters,
                    kernel_size=3,
                    strides=2,
                    padding='same')(h)

            h = LeakyReLU(0.2)(h)
            h = Conv2D(filters=filters,
                    kernel_size=3,
                    strides=1,
                    padding='same')(h)

            h = LeakyReLU(0.2)(h)

        for i in range(1):
            filters *= 2
            h = Conv2D(filters=filters,
                    kernel_size=3,
                    strides=2,
                    padding='same')(h)
            h = LeakyReLU(0.2)(h)
            h = Conv2D(filters=filters,
                    kernel_size=3,
                    strides=1,
                    padding='same')(h)
            h = LeakyReLU(0.2)(h)


        h_shape = K.int_shape(h)[1:]
        h = Flatten()(h)

        z_mean = Dense(latent_dim)(h) # p(z|x)的均值
        z_log_var = Dense(latent_dim)(h) # p(z|x)的方差


# adversarial network

        z = x
        z = Flatten()(z)
        z = Dense(1024,name='a1')(z)
        z = Dense(512,name='a2')(z)
        z = Dense(latent_dim,name='a3')(z)

        self.Advsior = Model(x,z)

        pertation = self.Advsior(x)

#Decoder



        z = Input(shape=(latent_dim,))
        h = z
        h = Dense(np.prod(h_shape))(h)
        h = Reshape(h_shape)(h)

        for i in range(2):
            h = Conv2DTranspose(filters=filters,
                                kernel_size=3,
                                strides=1,
                                padding='same')(h)
            h = LeakyReLU(0.2)(h)
            h = Conv2DTranspose(filters=filters,
                                kernel_size=3,
                                strides=2,
                                padding='same')(h)
            h = LeakyReLU(0.2)(h)
            filters //= 2

        x_recon = Conv2DTranspose(filters=1,
                                kernel_size=3,
                                activation='sigmoid',
                                padding='same')(h)

        self.decoder = Model(z, x_recon)



#clustering layer
        z = Input(shape=(latent_dim,))
        y = Dense(1024, activation='relu',name='c1')(z)
        # y = Lambda(GCN)(y)
        y = Dense(1024, activation='relu',name='c2')(y)
        # y = Lambda(GCN)(y)
        y = Dense(512, activation='relu',name='c3')(y)
        # y = Lambda(GCN)(y)
        y = Dense(num_classes, activation='softmax')(y)

        self.classfier = Model(z, y)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z_mean_1 = z_mean + pertation

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        z_1 = Lambda(sampling, output_shape=(latent_dim,))([z_mean_1, z_log_var])
        self.encoder = Model(x, z_mean)

        x_recon = self.decoder(z)
        x_recon1 = self.decoder(z_1)

        y = self.classfier(z_mean)
        y_1 = self.classfier(z_mean_1)

        gaussian = Gaussian(num_classes)
        z_prior_mean = gaussian(z)

        self.vae = Model(x, [x_recon,z_prior_mean,y])


# graph module

        W = costs.knn_affinity(z_mean, params['n_nbrs'], scale=1.97, scale_nbr=params['scale_nbr'])
        W = W - self.P
        layers = [
                  {'type': 'Orthonorm', 'name':'orthonorm'}
                  ]

        outputs = stack_layers(y,layers)
        Dy = costs.squared_distance(outputs)

        loss_SPNet =1* (K.sum(W * Dy))

# MIE
        def shuffling(x):
            idxs = K.arange(0, K.shape(x)[0])
            idxs = K.tf.random_shuffle(idxs)
            return K.gather(x, idxs)

        z_shuffle = Lambda(shuffling)(z_mean)
        z_z_1 = Concatenate()([z_mean, z_mean])
        z_z_2 = Concatenate()([z_mean, z_shuffle])

        z_in = Input(shape=(latent_dim * 2,))
        z1 = z_in
        z1 = Dense(latent_dim, activation='relu')(z1)
        z1 = Dense(latent_dim, activation='relu')(z1)
        z1 = Dense(latent_dim, activation='relu')(z1)
        z1 = Dense(1, activation='sigmoid')(z1)

        GlobalDiscriminator = Model(z_in, z1)

        z_z_1_scores = GlobalDiscriminator(z_z_1)
        z_z_2_scores = GlobalDiscriminator(z_z_2)
        global_info_loss = - K.mean(K.log(z_z_1_scores + 1e-6) + K.log(1 - z_z_2_scores + 1e-6))


#clustering network


        z_log_var = K.expand_dims(z_log_var, 1)

        lamb = 2.5  # 这是重构误差的权重，它的相反数就是重构方差，越大意味着方差越小。
        xent_loss = 1 * K.mean((x - x_recon) ** 2, 0)

        kl_loss = - 0.5 * (1 + z_log_var - K.square(K.expand_dims(z_mean, 1) - z_prior_mean) - K.exp(z_log_var))
        # kl_loss = - 0.5 * (z_log_var - K.square(z_prior_mean))
        kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0)
        cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)

        self.module_loss = lamb * K.sum(xent_loss)+1*K.sum(kl_loss)+1*K.sum(cat_loss)

# Attack loss


        selfloss = 1 * K.mean((z_mean_1 - z_mean) ** 2, 0)

        Dis = tf.diag_part(tf.matmul(y,y_1,transpose_b=True))

        xent1_loss = 1 * K.mean((x_recon - x_recon1) ** 2, 0)
        self.adv_loss = lamb * K.sum(xent1_loss) + 0.05 * K.sum(Dis) + 1 * K.sum(selfloss)


#defense


        z_in = Input(shape=(858,))
        z1 = z_in
        z1 = Dense(latent_dim, activation='relu')(z1)
        z1 = Dense(latent_dim, activation='relu')(z1)
        z1 = Dense(1, activation='sigmoid')(z1)

        self.Discriminator = Model(z_in, z1)


        c1 = tf.concat([tf.reshape(x_recon,[-1,784]),y],1)
        c2 = tf.concat([tf.reshape(x_recon1,[-1,784]),y_1],1)

        c1_shuffle = Lambda(shuffling)(c1)
        z_z_1 = Concatenate()([z_mean_1, c1])
        z_z_2 = Concatenate()([z_mean_1, c1_shuffle])

        z_z_1_scores = self.Discriminator(z_z_1)
        z_z_2_scores = self.Discriminator(z_z_2)
        info_loss_c1 = - K.mean(K.log(z_z_1_scores + 1e-6) + K.log(1 - z_z_2_scores + 1e-6))


        c2_shuffle = Lambda(shuffling)(c2)
        z_z_1 = Concatenate()([z_mean, c2])
        z_z_2 = Concatenate()([z_mean, c2_shuffle])

        z_z_1_scores = self.Discriminator(z_z_1)
        z_z_2_scores = self.Discriminator(z_z_2)
        info_loss_c2 = - K.mean(K.log(z_z_1_scores + 1e-6) + K.log(1 - z_z_2_scores + 1e-6))



        self.loss_defense = self.module_loss+2.5*(K.sum(info_loss_c1)+K.sum(info_loss_c2))




        self.D = K.mean(K.sum(tf.abs(z_mean - z_mean_1), 0)/K.sum(tf.abs(z_mean), 0))



        self.learning_rate = tf.Variable(0., name='spectral_net_learning_rate')
        self.train_step1 = tf.train.AdamOptimizer().minimize(self.loss_defense,var_list=[self.vae.weights,self.Discriminator.weights])
        self.train_step2 = tf.train.AdamOptimizer().minimize(self.adv_loss, var_list=self.Advsior.weights)
        K.get_session().run(tf.variables_initializer(self.vae.trainable_weights))

    def train_vae(self, x_train_unlabeled,x_dy,x_dy1,batch_size):
        # create handler for early stopping and learning rate scheduling

        losses,M = self.train_vae_step(
                return_var=[self.loss_defense],
                updates=[self.train_step1]+self.vae.updates+self.Discriminator.updates,
                x_unlabeled=x_train_unlabeled,
                inputs=self.x,
                x_dy=x_dy,
                x_dy1=x_dy1,
                batch_sizes=batch_size,
                batches_per_epoch=50)





        return losses,M

    def train_vae_step(self,return_var, updates, x_unlabeled, inputs,x_dy,x_dy1,
                   batch_sizes,
                   batches_per_epoch=100):

        return_vars_ = np.zeros(shape=(len(return_var)))

        # scale = get_scale(x_dy, 1000, 2)
        # train batches_per_epoch batches
        M = 0
        for batch_num in range(0, batches_per_epoch):
            feed_dict = {K.learning_phase(): 1}

            # feed corresponding input for each input_type

            batch_ids = np.random.choice(len(x_unlabeled), size=batch_sizes, replace=False)
            feed_dict[inputs] = x_unlabeled[batch_ids]
            feed_dict[self.Dy]=x_dy[batch_ids]
            feed_dict[self.Dy1] = x_dy1[batch_ids]


                        # feed_dict[P]=P[batch_ids]

            all_vars = return_var + updates
            all_vars_R,D =  K.get_session().run((all_vars,self.D), feed_dict=feed_dict)

            return_vars_ += np.asarray(all_vars_R[:len(return_var)])
            M  = M+D

            # M = K.get_session().run((self.centers_d),feed_dict=feed_dict)
            # print('confusion matrix{}: '.format(''))
            # print(np.round(centers_d, 2))
            # print(np.round(D, 4))
        return return_vars_,M




class Gaussian(Layer):

    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Gaussian, self).__init__(**kwargs)
    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='mean',
                                    shape=(self.num_classes, latent_dim),
                                    initializer='zeros')
    def call(self, inputs):
        z = inputs # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)
        return z * 0 + K.expand_dims(self.mean, 0)
    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, input_shape[-1])