import jax.numpy as jnp
class GELU:
    def __init__(self):
        pass
    def forward(self, x):
        self.x = x
        self.u = jnp.sqrt(2/jnp.pi) * (x + 0.044715 * x**3)
        self.tanh_u = jnp.tanh(self.u)
        return 0.5 * x * (1 + self.tanh_u)
    
    def backward(self, output_gradient):
        du_dx = jnp.sqrt(2/jnp.pi) * (1 + 3 * 0.044715 * self.x**2)
        df_dx = 0.5 * (1 + self.tanh_u) + 0.5 * self.x * (1 - self.tanh_u**2) * du_dx
        return output_gradient * df_dx
    
class Softmax :
    def __init__(self):
        pass

    def forward(self,input):
        self.input =input 
        input_max = jnp.max(self.input, axis=-1, keepdims=True)
        exps = jnp.exp(self.input-input_max)
        sum_exps = jnp.sum(exps, axis=-1, keepdims=True)
    
        # Return the output
        self.output = jnp.where(sum_exps == 0, 0.0, exps / sum_exps)
        return self.output
    
    def backward(self,output_gradient) :
        sum_output_gradient=jnp.sum(output_gradient*self.output,axis=-1,keepdims=True)
        return self.output *( output_gradient -sum_output_gradient)
