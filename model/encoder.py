from model.EncoderBlock import EncoderBlock
class Encoder :
    def __init__(self,d_model,adamw,offset,h,N): # N is the number of encoders to be used
        self.encoder_stack=[ EncoderBlock(d_model,adamw,h,offset=offset+i) for i in range(N)]

    def forward(self,x):
        for encoder in self.encoder_stack :
            x=encoder.forward(x)
        return x
    
    def backward(self,out_grad):
        for encoder in reversed(self.encoder_stack):
            input_grad=encoder.backward(out_grad)
            out_grad=input_grad
        return input_grad

