import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
from model.VIT import VIT
from data.dataloader import get_files,load_batch
from model.layers.optim.loss import CategoricalCrossEntropy
from model.layers.optim.adamw import AdamW

adamw = AdamW()
loss_fn = CategoricalCrossEntropy()

patch_size = 8 ;channels = 3 ;d_model = 256 ;batch_size = 64
seed = 42 ;N = 8 ;n = 64 ;h = 8 ;num_classes = 200

vit = VIT(patch_size,channels,adamw,d_model,batch_size,seed,N,h,n,num_classes)

data_dir = "/kaggle/input/tiny-imagenet/tiny-imagenet-200"
losses = []

files,lbls = get_files(data_dir,split='train')
num_batches = len(files)//batch_size
print(f"Found {len(files)} files | {num_batches} batches")

epochs = 60
t0 = time.time()

for epoch in range(epochs):
    ##shuffle each epoch
    idx = np.random.permutation(len(files))
    files = [files[i] for i in idx]
    lbls = [lbls[i] for i in idx]
    ep_start = time.time()
    print(f"Epoch {epoch+1}/{epochs} | {num_batches} batches")

    total_loss = 0.0 ;correct = 0 ;total = 0
    for b in range(num_batches):
        bf = files[b*batch_size:(b+1)*batch_size]
        bl = lbls[b*batch_size:(b+1)*batch_size]
        xb,yb = load_batch(bf,bl)

        y_oh = jax.nn.one_hot(yb,num_classes).reshape(-1,1,num_classes)
        adamw.step()
        out = vit.forward(xb)
        loss = loss_fn.forward(out,y_oh)
        grad = loss_fn.backward(out,y_oh)
        vit.backward(grad)

        total_loss += float(loss)
        preds = jnp.argmax(out,axis=-1).flatten()
        correct += int(jnp.sum(preds == yb))
        total += len(yb)

        if (b+1)%50==0:
            elapsed = time.time()-ep_start
            eta = (elapsed/(b+1))*(num_batches-(b+1))
            print(f"  Batch {b+1}/{num_batches} | Loss: {loss:.4f} | Acc: {correct/total:.4f} | Time: {elapsed:.0f}s | ETA: {eta/60:.1f}min")

    ep_time = time.time()-ep_start
    avg = total_loss/num_batches
    acc = correct/total
    losses.append(avg)
    hrs = (time.time()-t0)/3600
    print(f"Epoch {epoch+1} | Loss: {avg:.4f} | Acc: {acc:.4f} | Time: {ep_time/60:.1f}min | Total: {hrs:.1f}hrs")

    if (epoch+1)%5==0:
        with open(f"/kaggle/working/vit_e{epoch+1}.pkl","wb") as f:
            pickle.dump(vit,f)

with open("/kaggle/working/vit_final.pkl","wb") as f:
    pickle.dump(vit,f)
with open("/kaggle/working/losses.pkl","wb") as f:
    pickle.dump(losses,f)
