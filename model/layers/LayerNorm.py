import jax.numpy as jnp
class LayerNormalization:
    def __init__(self,d_model,adamW):
        self.d_model=d_model
        self.epsilon=1e-6


        self.gamma=jnp.ones((self.d_model,))
        self.beta=jnp.zeros((self.d_model,))
        self.adamW=adamW

    def forward(self,x):
        self.x=x
        mu=jnp.mean(self.x,axis=-1,keepdims=True) ##average over last dim and keepdim=1 for broadcasting 
        self.sigma_square=jnp.mean(jnp.square(self.x-mu),axis=-1,keepdims=True)
        self.x_norm=(self.x-mu)/jnp.sqrt(self.sigma_square+self.epsilon) 
        
        return self.gamma *self.x_norm +self.beta
    
    def backward(self,output_gradient):
       #dL/dgamma=dL/dy *dy/dgamma
       d_gamma=jnp.sum(output_gradient * self.x_norm,axis=(0,1))

       #dL/dbeta=dL/dy *dY/dbeta
       d_beta=jnp.sum(output_gradient,axis=(0,1))

        
        
       input_gradient = (1 / jnp.sqrt(self.sigma_square + self.epsilon)) * ((output_gradient * self.gamma) -
                                                                             (jnp.sum(output_gradient * self.gamma, axis=-1, keepdims=True) / self.d_model) - 
                                                        (self.x_norm / self.d_model) * jnp.sum(output_gradient * self.gamma * self.x_norm, axis=-1, keepdims=True))
       #update gamma
       self.gamma=self.adamW.update(f"{id(self)}_layer_norm_gamma",self.gamma,d_gamma)


       #update beta
       self.beta=self.adamW.update(f"{id(self)}_layer_norm_beta",self.beta,d_beta) 

       
       

       return input_gradient
    