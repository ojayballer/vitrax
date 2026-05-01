import jax.numpy as jnp 
import jax

###learned position embedding
class PositionEmbedding :
    def __init__(self,N,d_model,seed,adamW):
        self.N=N
        self.d_model=d_model
        key=jax.random.PRNGKey(seed)
        
        self.adamw=adamW

        #xavier-Glorot initialisation
        std=jnp.sqrt(2/(self.N+self.d_model))
        
        self.weights=jax.random.normal(key,(N+1,d_model))* std # N+1 becuse of the CLS token that was concatenated
    

    def forward(self,patch_embedding):  # for the forwrad pass,we want to add patch embedding to the learned positional embeddings
        return patch_embedding +self.weights
    

    def backward(self,output_gradient):

        # update self.weights using adamw
        self.weights = self.adamw.update(f"{id(self)}_pos_embed", self.weights, output_gradient)
        return output_gradient  ## patch embedding grad

