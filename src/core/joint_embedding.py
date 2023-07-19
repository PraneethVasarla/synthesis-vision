import sys
import os
import concurrent.futures
from tqdm.auto import tqdm
import tensorflow as tf

sys.path.append(os.getcwd())

from src.embeddings.text_embeddings import get_text_embedding
from src.embeddings.image_embeddings import get_image_embeddings
from src.embeddings.fusion import FusionWithAttention

class JointEmbedding:
    def __init__(self,text_model,tokenizer,vision_model,feature_extractor):
        self.image_embeddings = None
        self.text_embeddings = None
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.vision_model = vision_model
        self.feature_extractor = feature_extractor

    def get_pair_embeddings(self, images, captions):
        # Create separate threads for images and texts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            image_future = executor.submit(get_image_embeddings, images_names=images, model=self.vision_model,
                                           feature_extractor=self.feature_extractor)
            text_future = executor.submit(get_text_embedding, captions, model=self.text_model, tokenizer=self.tokenizer)

            # Wait for all threads to finish and disable the threaded context manager
            futures = [image_future, text_future]
            with tqdm(total=len(futures), desc="Processing pairs", dynamic_ncols=True) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)

            # Retrieve the results of the tasks
            self.image_embeddings = image_future.result()
            self.text_embeddings = text_future.result()

    def fuse_embeddings_with_attention(self):
        text_embeddings = tf.convert_to_tensor(self.text_embeddings, dtype=tf.float32)
        image_embeddings = tf.convert_to_tensor(self.image_embeddings, dtype=tf.float32)

        # Use the FusionWithAttention class imported from the fusion.py file
        text_dim = text_embeddings.shape[-1]
        image_dim = image_embeddings.shape[-1]
        hidden_dim = 768  # You can adjust this dimension according to your needs

        # Create the fusion model and apply attention-based fusion
        fusion_model = FusionWithAttention(text_dim, image_dim, hidden_dim)

        # Fuse each text and image embedding pair independently
        fused_embeddings = []
        for text_emb, image_emb in tqdm(zip(text_embeddings, image_embeddings),desc="Fusing Embeddings"):
            fused_embedding = fusion_model(text_emb, image_emb)

            # Apply max-pooling over the image embeddings to reduce dimension from (197, hidden_dim) to (hidden_dim,)
            pooled_embedding = tf.reduce_max(fused_embedding, axis=0)  # Max-pooling over the image embeddings
            fused_embeddings.append(tf.expand_dims(pooled_embedding, axis=0))

        # Stack the fused embeddings into a single tensor
        fused_embeddings = tf.concat(fused_embeddings, axis=0)

        return fused_embeddings