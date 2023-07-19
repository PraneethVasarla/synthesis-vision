import tensorflow as tf

class FusionWithAttention(tf.keras.layers.Layer):
    def __init__(self, text_dim, image_dim, hidden_dim):
        super(FusionWithAttention, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim

        # Text projection layer
        self.text_projection = tf.keras.layers.Dense(hidden_dim, activation='relu')

        # Image projection layer
        self.image_projection = tf.keras.layers.Dense(hidden_dim, activation='relu')

        # Attention layer
        self.attention = tf.keras.layers.Attention()

        # Fusion layer
        self.fusion_linear = tf.keras.layers.Dense(hidden_dim, activation='relu')

    def call(self, text_embeddings, image_embeddings):
        # Project the text and image embeddings to the hidden dimension
        projected_text = self.text_projection(text_embeddings)  # Shape: (batch_size, hidden_dim)
        projected_image = self.image_projection(image_embeddings)  # Shape: (batch_size, hidden_dim)

        # Add an extra dimension to image_embeddings to make it compatible with attention mechanism
        expanded_image_embeddings = tf.expand_dims(projected_image, axis=1)  # Shape: (batch_size, 1, hidden_dim)

        # Apply attention mechanism to fuse the embeddings
        fused_embedding = self.attention([projected_text, expanded_image_embeddings])

        # Apply the fusion linear layer to further process the fused embeddings
        fused_embedding = self.fusion_linear(fused_embedding)

        return fused_embedding
