import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras import layers, initializers


class Encoder(layers.Layer):
    """ 
    Encodes input data tensor to latent triple vector (z_mean, z_log_var, z)
    """

    def __init__(self, latent_dim: int, name:str = "encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_mean = layers.Dense(latent_dim) # linear activations
        self.dense_log_var = layers.Dense(latent_dim, kernel_initializer = initializers.RandomNormal(stddev=0.1))
        self.sampling = Sampling()

    def call(self, inputs: tf.Tensor):
        x = inputs
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling.reparametrization((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """
    Converts z, the encoded tensor, back into a reconstructed term.
    """

    def __init__(
        self,
        original_dim: int,
        distribution: str = 'default',
        name: str = 'decoder'
        , **kwargs
    ):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.distribution = distribution
        self.sampling = Sampling()

        # simple relu decoder
        self.dense_output = layers.Dense(original_dim, activation = 'relu')
        
        # mean and logvar for normal
        self.dense_mean = layers.Dense(original_dim, bias_initializer='ones') # linear activations
        self.dense_log_var = layers.Dense(original_dim, kernel_initializer = initializers.RandomNormal(stddev=0.1)) # use logvar as var need constaint to be positive
        
        # bernoulli decoder
        self.bernoulli_lambda = layers.Dense(original_dim, activation = 'sigmoid')

        # gamma decoder
        self.gamma_log_alpha = layers.Dense(original_dim, bias_initializer='ones', kernel_initializer = initializers.RandomNormal(stddev=0.1)) # use logvar as var need constaint to be positive
        self.gamma_log_beta = layers.Dense(original_dim, bias_initializer='zeros', kernel_initializer = initializers.RandomNormal(stddev=0.1)) # use logvar as var need constaint to be positive
        
    def call(self, inputs: tf.Tensor):
        if self.distribution == 'default':
            return self.dense_output(inputs)

        elif self.distribution == 'normal':
            z = inputs
            mean = self.dense_mean(z)
            log_var = self.dense_log_var(z)
            output = self.sampling.normal((mean,log_var))
            return output

        elif self.distribution == 'bernoulli':
            z = inputs
            mean = self.bernoulli_lambda(z)
            output = self.sampling.bernoulli(mean)
            output = tf.cast(output, tf.float32) 
            return output
            
        elif self.distribution == 'gamma':
            z = inputs
            log_alpha = self.gamma_log_alpha(z)
            log_beta = self.gamma_log_beta(z)
            output = self.sampling.gamma((log_alpha,log_beta))
            return output


class VariationalAutoEncoder(Model):
    """
    Combines the encoder and decoder into one model for training.
    """

    def __init__(
        self,\
        original_dim: int,\
        latent_dim: int,\
        distribution: str,\
        name:str = "variational autoencoder",\
        **kwargs\
    ):
        super(VariationalAutoEncoder, self).__init__(name = name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim = latent_dim)
        self.decoder = Decoder(original_dim, distribution = distribution)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs) 
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed


class Factory:
    '''
    Trains VAE model and generates synthetic data.

    Initialize: Only 'data_spec' is mandatory - specifies which columns are categorical and which continuous.
    
    Methods: 
    learn(data: pd.DataFrame): Trains VAE model for each category. Models are saved in self.models.
                              Plots  for each category a fitted sample vs real data.
                              
    generate(num_rows: int, data:pd.DataFrame) generates synthetic data points using VAE models from self.models.
    '''

    def __init__(
        self, data_spec: dict, \
        distribution_spec: dict = {},\
        log_transform:bool = True,\
        latent_dim:int = 16,\
        batch_size:int = 64,\
        epochs:int = 15,\
        learning_rate:float =1e-3
    ):
        self.val_cols = data_spec.get('val_cols',[])
        self.cat_col = data_spec.get('cat_col',[])
        self.distribution_spec = distribution_spec
        self.log_transform = log_transform
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def learn(self, data: pd.DataFrame):
        '''
        For each category we train separate VAE. self.model will save all VAEs
        '''
        self.model = {}
        categories = data[self.cat_col[0]].drop_duplicates()
        for category_name in categories:
            # real data per category
            df_cat = data[data[self.cat_col[0]]==category_name]
            # real data per category only value columns
            df_cat_val = df_cat[self.val_cols]
            
            if self.log_transform and self.distribution_spec.get(category_name,None)!='bernoulli':
                for value_col in self.val_cols:
                    df_cat_val[value_col] =  np.log(df_cat_val[value_col] + 1)
            
            # dimension of real data we want to synthesize
            original_dim = len(df_cat_val.columns)

            # VAE OBJECT
            vae = VariationalAutoEncoder(
                    original_dim,\
                    latent_dim = self.latent_dim,\
                    distribution = self.distribution_spec.get(category_name,'default')
                )
            
            # loss and optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
            mse_loss_fn = tf.keras.losses.MeanSquaredError()
            loss_metric = tf.keras.metrics.Mean()
            
            # prepare data in form of batches of tensors
            train_dataset = tf.data.Dataset.from_tensor_slices(df_cat_val)
            train_dataset = train_dataset.shuffle(buffer_size=len(df_cat_val)).batch(self.batch_size)
            
            # Iterate over epochs.
            for epoch in range(self.epochs):
                ls = []
                print(f"Start of epoch {epoch}")
                for step, x_batch_train in enumerate(train_dataset):
                    with tf.GradientTape() as tape:
                        # goes into call method
                        reconstructed = vae(x_batch_train) 
                        
                        ls.extend(list(reconstructed.numpy().flatten()))
                        # Compute reconstruction loss
                        loss = mse_loss_fn(x_batch_train, reconstructed)
                        loss += sum(vae.losses)  # Add KL divergence term (entropy)
                    grads = tape.gradient(loss, vae.trainable_weights)
                    
                    optimizer.apply_gradients(zip(grads, vae.trainable_weights))
                    loss_metric(loss)
                    
                # final loss is likelihood-ELBO
                loss_metric(loss)
                print('Loss:', loss_metric.result().numpy())


            # Plot comparison between fitted data and real after the last epoch
            synthetic = pd.DataFrame()
            if self.log_transform  and self.distribution_spec.get(category_name,None)!='bernoulli':
                synthetic['mIncome'] = np.exp(pd.Series(ls)) -1 # exponentiate back
            else:
                synthetic['mIncome'] = pd.Series(ls)

            # data
            clip = df_cat[(df_cat['MonthlyIncome']<30000) & (df_cat['MonthlyIncome']>0)]
            clip_synthetic = synthetic[synthetic['mIncome']<30000]
            # plot
            fig = plt.figure(figsize=(9,5))
            sns.histplot(clip, x = 'MonthlyIncome', element="poly",  stat = 'density',alpha = 1)
            sns.histplot(clip_synthetic, x = 'mIncome', element="poly",  stat = 'density', color= 'purple', alpha = 0.5)
            fig.legend(labels=['Real','Synthetic'])
            print(f'FINISHED {category_name}')
            plt.title(f'Last Epoch, Effect category {category_name}')
            plt.show()
                
            # SAVE MODEL for generating synthetic samples
            self.model[category_name] = vae
            
            
    def generate(self, num_rows: int, data:pd.DataFrame):
        
        # obtain real data sample - we use it to get the categories for which we will generate data
        sample = data.sample(min(num_rows,len(data))).reset_index()
        num_rows -= len(data)
        while num_rows > 0:
            sample = pd.concat([data.sample(min(num_rows,len(data))).reset_index(),sample])
            num_rows -= len(data)
        
        categories = sample[self.cat_col[0]].drop_duplicates()
        
        ls = [] # holds synthesized frames for each category
        # sample from learned  VAE model for each category
        for category_name in categories:
            df_cat = sample[sample[self.cat_col[0]]==category_name]
            df_cat_val = df_cat[self.val_cols]
            
            # log transform
            if self.log_transform and self.distribution_spec.get(category_name,None)!='bernoulli':
                for value_col in self.val_cols:
                    df_cat_val[value_col] =  np.log(df_cat_val[value_col] + 1)

            train_dataset = tf.data.Dataset.from_tensor_slices(df_cat_val)
            train_dataset = train_dataset.shuffle(buffer_size=len(df_cat_val)).batch(len(df_cat_val))

            reconstructed = pd.DataFrame(self.model[category_name](tuple(train_dataset)[0]).numpy())
            reconstructed.columns = self.val_cols
            
            # exponentiate back
            if self.log_transform and self.distribution_spec.get(category_name,None)!='bernoulli':
                for value_col in self.val_cols:
                    reconstructed[value_col] = np.exp(reconstructed[value_col]) - 1
                    
            # add category column
            reconstructed[self.cat_col[0]] = df_cat[self.cat_col[0]].iloc[0]
            # save reconstructed dataframe
            ls.append(reconstructed)
            
        return pd.concat(ls)
    
class Sampling(layers.Layer):
    """
    reparametrization trick uses (z_mean, z_log_var) to sample z
    normal, bernoulli, gamma sampling with tensors
    """

    def reparametrization(self, inputs: 'tuple[tf.Tensor]') -> tf.Tensor :
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0] # batch size
        dim = tf.shape(z_mean)[1] # latent dimension
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim)) # tensor with normal distribution of values.
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon # reparametrization trick

    def normal(self, inputs: 'tuple[tf.Tensor]') -> tf.Tensor :
        mean, log_var = inputs
        output = tfp.distributions.Normal(mean,tf.sqrt(tf.exp(log_var))).sample(1)[0]
        return output 

    def bernoulli(self, inputs: tf.Tensor) -> tf.Tensor :
        mean = inputs
        output = tfp.distributions.Bernoulli(mean).sample(1)[0]
        return output 

    def gamma(self, inputs: 'tuple[tf.Tensor]') -> tf.Tensor :
        log_alpha, log_beta = inputs
        output = tfp.distributions.Gamma(tf.exp(log_alpha),tf.exp(log_beta)).sample(1)[0]
        return output 
