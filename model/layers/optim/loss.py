import jax.numpy as jnp 
class CategoricalCrossEntropy():
    def forward(self, y_pred, y_true):
        y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
        # Sum across classes (axis=-1)
        return jnp.mean(-jnp.sum(y_true * jnp.log(y_pred), axis=-1))

    def backward(self, y_pred, y_true):
        y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
        return -y_true / y_pred
