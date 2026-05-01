import jax.numpy as jnp
from model.layers.dense import Dense
class PatchEmbedding :
    def __init__(self,patch_size,channels,d_model,seed,adamw):   #(batch_size,H,W,C)  ,patch size
       
        self.patch_size=patch_size
        self.channels=channels
        
       
        #create the linear projection object
        self.dense=Dense(self.patch_size**2*self.channels,d_model,adamw,seed) #input size,output_size,weight takes the shape of (input size,output_size)
    
    def forward(self,image):

        self.batch_size=image.shape[0]
        self.height=image.shape[1]
        self.width=image.shape[2]
       
         ##now i want to rreshape the image into equal sized patches and then flattn the dimension of each of those patches
         #first we have to reshape the image to (batch,H//P,P,W//P,P,channels) 
        self.patches= jnp.reshape(image,(self.batch_size,self.height//self.patch_size,self.patch_size,self.width//self.patch_size,self.patch_size,self.channels))

        #now we transpose to shape (batch,H//P,W//P,P,P,C) 
        self.patches=jnp.transpose(self.patches,(0,1,3,2,4,5))
        
        #finally , we reshape it to (batch,N,P*P*C) , ##N = HW/P^2,N is the number of equal sized patches an image can be splitted into 
        self.patches=jnp.reshape(self.patches,(self.batch_size,self.height//self.patch_size * self.width//self.patch_size,self.patch_size**2 *self.channels))

        #we want to pass each patch into a linear projection(dense layer)
        return self.dense.forward(self.patches)  # multiply the patches by the weights and add bias 
         
         ##output shape = (batch,N,d_model)

    def backward(self,output_gradient):
        return self.dense.backward(output_gradient)
        


