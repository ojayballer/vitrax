# Vision Transformer from Scratch

A Vision Transformer trained on Tiny ImageNet (200 classes), built entirely with `jax.numpy`. No autograd. Every gradient through every layer, multi-head self-attention, patch embeddings, layer normalization, the full encoder stack, all hand derived and implemented manually.

## what is this

This is a complete implementation of the Vision Transformer from [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020). The model takes a 64x64 RGB image, splits it into a grid of 8x8 patches, projects each patch into a 256 dimensional embedding, prepends a learnable CLS token, adds positional embeddings, and passes the resulting sequence through 8 Transformer encoder blocks. The CLS token output is then projected to 200 classes via a dense head.

I wanted to understand what happens when you strip away all the engineering conveniences and force yourself to derive every backward pass by hand. Not because it is practical, but because there is no faster way to learn how these things actually work. When your gradients are wrong, nothing converges. When they are right, you know it because you proved it.

All gradients were numerically verified against finite differences to `1e-11` precision.

## architecture

<p align="center">
  <img src="assets/vit_architecture.png" width="800"/>
</p>

<p align="center"><sub>Figure 1 from Dosovitskiy et al. (2020)</sub></p>

Config: `patch_size=8`, `channels=3`, `d_model=256`, `heads=8`, `layers=8`, `seq_len=64`, `num_classes=200`, `batch_size=64`

Each encoder block contains multi-head self-attention with 8 heads (d_k=32), a two layer feed-forward network (256 to 1024 to 256 with GELU), and post-norm residual connections. Positional embeddings are learned, not sinusoidal. The classification head is a single dense layer on the CLS token.

Every component has a hand written backward pass. The attention backward computes gradients through the softmax Jacobian vector product, through the scaled dot product, and back through the Q/K/V projections. The layer norm backward handles the full chain through variance and mean. Patch embedding backward flows through the linear projection back to image space.

Total parameters: ~6.4M

## results

Trained on Tiny ImageNet (100k images, 200 classes, 64x64 RGB) on a Kaggle P100. No data augmentation, no dropout, no learning rate scheduling. Just the raw architecture with AdamW.

| | |
|---|---|
| Best Validation Accuracy | **21.84%** |
| Random Chance | 0.50% (1/200) |

ViTs lack the inductive biases that CNNs have. Convolutions bake in translation invariance and locality, meaning the network already "knows" that nearby pixels matter before it sees a single training example. Transformers have none of that. Every spatial relationship has to be learned from scratch, purely from data. The original ViT paper needed JFT-300M (300 million images) to reach competitive performance with CNNs. This model was trained on 100k images with no data augmentation, no dropout, and no learning rate scheduling, which means it had to discover both what features matter and where they are in the image, all from a relatively small dataset with nothing to prevent it from memorizing.

The train vs validation curve tells the story. The model picks up real features early on, then starts memorizing the training set as capacity outpaces the data. Classic overfitting when you have a high capacity model with no data augmentation or dropout to keep it honest.

<p align="center">
  <img src="assets/train_vs_val.png" width="600"/>
</p>

<p align="center">
  <img src="assets/training_loss.png" width="600"/>
</p>

## what would improve it

Adding dropout (even 0.1) and basic data augmentation (random horizontal flips, random crops) would likely push validation accuracy above 30%. Learning rate warmup with cosine decay would help early training stability. These are well understood techniques that I intentionally left out to keep the implementation purely about the architecture itself.

## project structure

```
.
├── model/
│   ├── VIT.py                          # full ViT: patch embed + position embed + encoder + classification head
│   ├── encoder.py                      # stacks N encoder blocks
│   ├── EncoderBlock.py                 # MHA + FFN + residual + layer norm
│   └── layers/
│       ├── multiheadAttention.py       # scaled dot product attention, 8 heads
│       ├── FeedForward.py              # two layer FFN with GELU
│       ├── dense.py                    # linear projection with manual backward
│       ├── LayerNorm.py                # layer normalization with manual backward
│       ├── Activation.py               # GELU activation
│       ├── PatchEmbedding.py           # image to patch sequence
│       ├── PositionEmbedding.py        # learned positional embeddings + CLS token
│       └── optim/
│           ├── adamw.py                # AdamW optimizer from scratch
│           └── loss.py                 # categorical cross entropy
├── data/
│   └── dataloader.py                   # loads Tiny ImageNet, batches on the fly
├── train.py                            # training loop with checkpointing
├── evaluate.py                         # validation evaluation
├── configs.yml                         # hyperparameters
└── requirements.txt
```

## usage

```bash
pip install jax jaxlib pillow numpy
```

Training (expects Tiny ImageNet at `tiny-imagenet-200/`):
```bash
python train.py
```

Evaluation:
```bash
python evaluate.py
```

Download Tiny ImageNet from http://cs231n.stanford.edu/tiny-imagenet-200.zip

## references

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Helling, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.

Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*.

## license

MIT
