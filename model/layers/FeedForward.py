from model.layers.dense import Dense
from model.layers.Activation import GELU
class FeedForward :
    def __init__(self,d_model,offset,adamw):
        self.d_model=d_model
        self.d_diff=4*self.d_model ##expand to 4d

        self.gelu=GELU()
        self.dense1=Dense(self.d_model,self.d_diff,adamw,seed=offset+1)
        self.dense2=Dense(self.d_diff,self.d_model,adamw,seed=offset+3)

    def forward(self,x):
        x=self.dense2.forward(self.gelu.forward(self.dense1.forward(x)))
        return x
    
    def backward(self,output_gradient):
        d2_grad=self.dense2.backward(output_gradient)
        gelu_back=self.gelu.backward(d2_grad)

        ##input grad 
        return self.dense1.backward(gelu_back)
        