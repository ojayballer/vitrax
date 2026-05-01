import sys
import jax.numpy as jnp
import numpy as np
import pickle
from data.dataloader import get_files,load_batch

data_dir = "tiny-imagenet"
batch_size = 64

##load best checkpoint
with open("vit_e10.pkl","rb") as f:
    vit = pickle.load(f)

##run validation
files,lbls = get_files(data_dir,split='val')
num_batches = len(files)//batch_size
print(f"Val: {len(files)} files | {num_batches} batches")

correct = 0
total = 0
for b in range(num_batches):
    bf = files[b*batch_size:(b+1)*batch_size]
    bl = lbls[b*batch_size:(b+1)*batch_size]
    xb,yb = load_batch(bf,bl)
    out = vit.forward(xb)
    preds = jnp.argmax(out,axis=-1).flatten()
    correct += int(jnp.sum(preds == yb))
    total += len(yb)

acc = correct/total
print(f"Val Accuracy: {acc*100:.2f}%")
