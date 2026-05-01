from model.layers.LayerNorm import LayerNormalization
from model.layers.FeedForward  import FeedForward
from model.layers.multiheadAttention import MultiHeadAttention

class EncoderBlock :
    def __init__(self,d_model,adamw,h,offset):#n is the number of heads for multi-head attention
        self.norm1=LayerNormalization(d_model,adamw)
        self.norm2=LayerNormalization(d_model,adamw)
        ####mha
        self.mha=MultiHeadAttention(d_model,h,adamw,offset=offset)
        self.ffw=FeedForward(d_model,offset=offset+1,adamw=adamw)

    def forward(self,x): ##embedded patches
         output=self.norm1.forward(x)   ##pre-normalization
         mha_fw=self.mha.forward(output)
         output_1=mha_fw+x
         output_2=self.norm2.forward(output_1) #pre-norm
         output_2=self.ffw.forward(output_2)
         res=output_1+ output_2
         return res
    
    def backward(self,output_grad):
        output_1_grad=output_grad ;output_2_grad=output_grad
        output_2_grad=self.ffw.backward(output_2_grad)
        output_1_grad=self.norm2.backward(output_2_grad) + output_1_grad #total output_1_grad
        mha_fw_grad=output_1_grad ;x_grad_1=output_1_grad
        out_grad=self.mha.backward(mha_fw_grad)
        x_grad_2=self.norm1.backward(out_grad)

        ##accumulate input grad
        return x_grad_1+x_grad_2
    
       

