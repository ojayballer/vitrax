from model.encoder import Encoder
from model.layers.PatchEmbedding import PatchEmbedding
from model.layers.PositionEmbedding import PositionEmbedding
import jax 
import jax.numpy as jnp
from model.layers.dense import Dense
from model.layers.Activation import Softmax
from model.layers.LayerNorm import LayerNormalization
class VIT :
    def __init__(self,patch_size,channels,adamw,d_model,batch_size,seed,N,h,n,num_classes):  #N is the number of encoder blocks,n is the numnber of patches
        #h is the number of heads for mha
        self.N=N
        self.n=n
        self.d_model=d_model
        self.batch_size=batch_size
        self.pe=PatchEmbedding(patch_size,channels,d_model,seed,adamw)
        self.pos_em=PositionEmbedding(n,d_model,seed,adamw)
        key=jax.random.PRNGKey(seed)
        self.cls_token=jax.random.normal(key,(self.batch_size,1,d_model))
        self.enc=Encoder(d_model,adamw,offset=seed,h=h,N=N)

        ##layer norm
        self.norm=LayerNormalization(d_model,adamw)

        ##final MLP head
        self.mlp_head=Dense(d_model,num_classes,adamw,seed=seed)
        #softmax
        self.softmax=Softmax()
        
       

    def forward(self,x):
        self.patch_embedding=self.pe.forward(x)#
       
        self.patch_embedding=jnp.concatenate([self.cls_token,self.patch_embedding],axis=1)  ##(batch,N+1,d_model)
        self.pos_embedding=self.pos_em.forward(self.patch_embedding) #(batch,N+1,d_model)
        encoder_output=self.enc.forward(self.pos_embedding)


        ##now we have to pass that output into a final MLP head for classification
         
         #but first ,we apply layer norm 
        cls_output=self.norm.forward(encoder_output[:,0:1,:] )#(batch,d_model)

        return self.softmax.forward(self.mlp_head.forward(cls_output))  #(batch,d_diff) where d_diff is the number of classes
    
    def backward(self, out_grad):
        out_grad = self.softmax.backward(out_grad)
        cls_output_grad = self.mlp_head.backward(out_grad)
        cls_output_grad = self.norm.backward(cls_output_grad)

        encoder_output_grad = jnp.zeros((self.batch_size, self.n+1, self.d_model))
        encoder_output_grad = encoder_output_grad.at[:, 0:1, :].set(cls_output_grad)
 
        pos_embedding_grad = self.enc.backward(encoder_output_grad)
        patch_embedding_grad = self.pos_em.backward(pos_embedding_grad)
        patch_embedding_grad = patch_embedding_grad[:, 1:, :]

        
        return self.pe.backward(patch_embedding_grad)

