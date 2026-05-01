import jax.numpy as jnp
import jax
class Dense :
    def __init__(self,d_model,d_diff,adamw,seed):
        self.d_model=d_model ;self.d_diff=d_diff
        key=jax.random.PRNGKey(seed)
        
        #xavier initialization 
        std= jnp.sqrt(2/(self.d_model+self.d_diff))

        self.weights=jax.random.normal(key,(self.d_model,self.d_diff))*std
        self.bias=jnp.zeros((self.d_diff,))

        self.adamw=adamw
    
    def forward(self,x):
        self.x=x
        return self.x @ self.weights +self.bias  ## (batch,N,d_diff)
    
    def backward(self,output_gradient): 
        #dl/dw
        weights_gradient=jnp.sum(self.x.transpose(0,2,1) @ output_gradient ,axis=0)   ##sum over batch

        #dl/dx
        input_gradient=  output_gradient @ self.weights.transpose(1,0)   ## (d_diff,d_model)*(batch,N,d_diff) -->(batch,N,d_model)

        #dl/db 
        bias_gradient =jnp.sum(output_gradient,axis=(0,1))  ### sum over batch and N 
       
        self.weights=self.adamw.update(f"{id(self)}_weights",self.weights,weights_gradient)
        self.bias=self.adamw.update(f"{id(self)}_bias",self.bias,bias_gradient)

        return input_gradient
    
        
         