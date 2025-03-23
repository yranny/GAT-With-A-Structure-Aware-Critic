import tensorflow as tf
from tensorflow.keras import layers
from spektral.layers import GATConv

class VQAGATModel(tf.keras.Model):
    def __init__(self, n_classes, hidden_dim=16, attn_heads=4):
        super(VQAGATModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads

        self.gat1 = GATConv(hidden_dim, attn_heads=attn_heads, activation='elu')
        self.dropout = layers.Dropout(0.2)
        self.gat2 = GATConv(n_classes, attn_heads=1, activation='softmax')

        # Match layout to GAT output size
        self.layout_dense = layers.Dense(hidden_dim * attn_heads, activation='relu')
        self.fusion = layers.Add()

    def call(self, inputs, training=False):
        x, a, layout = inputs  # Node features, adjacency, symbolic layout

        x1 = self.gat1([x, a])
        x1 = self.dropout(x1, training=training)

        layout_emb = self.layout_dense(layout)
        x1_guided = self.fusion([x1, layout_emb])  # Guided feature fusion

        out = self.gat2([x1_guided, a])
        return out

    def get_attention_weights(self):
        return self.gat1.attn_coeff

