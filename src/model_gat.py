import tensorflow as tf
from spektral.layers import GATConv

class GATModel(tf.keras.Model):
    def __init__(self, n_classes, hidden_dim=32):
        super().__init__()
        self.gat1 = GATConv(hidden_dim, activation='elu', attn_heads=4)
        self.gat2 = GATConv(n_classes, activation='softmax', attn_heads=1)

    def call(self, inputs):
        x, a = inputs
        x = self.gat1([x, a])
        self.attn_weights = self.gat1.attn_coefficients  # Store attention for analysis
        return self.gat2([x, a])

