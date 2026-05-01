from model.layers.dense import Dense
import jax.numpy as jnp
from model.layers.Activation import Softmax
class MultiHeadAttention :
    def __init__(self,d_model,n,adamw,offset): #n is the number of heads for multi-head attention

        #create dense layers for Q,K and V
        self.WQ=Dense(d_model,d_model,adamw,seed=offset+2) 
        self.WK=Dense(d_model,d_model,adamw,seed=offset+3)
        self.WV=Dense(d_model,d_model,adamw,seed=offset+7)
        
        self.d_model=d_model
        self.n=n
        self.d_k=self.d_model//n

        #create softmax object 
        self.softmax=Softmax()

        
        #dense layer for mha
        self.d=Dense(d_model,d_model,adamw,seed=offset+4)
    

    def forward(self,x): #this same input goes into the Q,K and V
        self.Q=self.WQ.forward(x)  #(batch,N,d_model)
        self.K=self.WK.forward(x)
        self.V=self.WV.forward(x)

        #reshape (batch,N,d_model) to (batch,N,n,d_k) and then transpose to (batch,n,N,d_k)
        self.Q_heads=jnp.reshape(self.Q,(self.Q.shape[0],self.Q.shape[1],self.n,self.d_k)).transpose(0,2,1,3)
        self.K_heads=jnp.reshape(self.K,(self.K.shape[0],self.K.shape[1],self.n,self.d_k)).transpose(0,2,1,3)
        self.V_heads=jnp.reshape(self.V,(self.K.shape[0],self.K.shape[1],self.n,self.d_k)).transpose(0,2,1,3)



        #perform attention on every pairs of heads
        self.scores=self.Q_heads@ self.K_heads.transpose(0,1,3,2)/jnp.sqrt(self.d_k) #(batch,n,N,N)
        self.attention_weights= self.softmax.forward(self.scores)
        self.attention=self.attention_weights @ self.V_heads  #(batch,n,N,d_k) 


        #===================multi-head-attention==========#######
        #======================================================================================================#######

        #transpose (batch,n,N,d_k) back to (batch,N,n,d_k)
        self.attention=jnp.transpose(self.attention,(0,2,1,3))

        ##mha
        self.multiHeadAttention= jnp.reshape(self.attention,(self.attention.shape[0],self.attention.shape[1],self.d_model))

        #finally,project mha into a dense layer
        return self.d.forward(self.multiHeadAttention)


     
    def backward(self,output_gradient):
       
        mha_dense=self.d.backward(output_gradient)

        attention_grad=jnp.reshape(mha_dense,(self.Q.shape[0],self.Q.shape[1],self.n,self.d_k)).transpose(0,2,1,3)
        
        d_attention=attention_grad @ self.V_heads.transpose(0,1,3,2)
        self.V_heads_backward=self.attention_weights.transpose(0,1,3,2) @ attention_grad 


        d_scores=self.softmax.backward(d_attention)

        
        self.Q_heads_backward=d_scores @ self.K_heads/jnp.sqrt(self.d_k) #(K^T)^T =K
        self.K_heads_backward=(self.Q_heads.transpose(0,1,3,2) @ d_scores).transpose(0,1,3,2) /jnp.sqrt(self.d_k)


        #===========reshape back to (batch,N,d_model)================##########
        #=============================================================#####

        ##V
        self.V_heads_backward = jnp.reshape(jnp.transpose(self.V_heads_backward, (0,2,1,3)), (self.V.shape[0], self.V.shape[1], self.d_model))

        ##Q
        self.Q_heads_backward = jnp.reshape(jnp.transpose(self.Q_heads_backward, (0,2,1,3)), (self.Q.shape[0], self.Q.shape[1], self.d_model))

        ##K
        self.K_heads_backward = jnp.reshape(jnp.transpose(self.K_heads_backward, (0,2,1,3)), (self.K.shape[0], self.K.shape[1], self.d_model))

        #compute partial input gradients

        Q_input_grad=self.WQ.backward(self.Q_heads_backward)
        K_input_grad=self.WK.backward(self.K_heads_backward)
        V_input_grad=self.WV.backward(self.V_heads_backward)


        #accumulate total input gradient
        return Q_input_grad+ K_input_grad + V_input_grad
    



        


        