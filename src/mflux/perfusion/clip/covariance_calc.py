import os

import mlx.core as mx
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from mflux.models.text_encoder.clip_encoder.clip_text_model import CLIPTextModel
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.tokenizer_handler import TokenizerHandler


def download_or_get_cached_dataset():
    return load_dataset("ytaek-oh/laioncoco-subset-100k")


# Helper to load tokenizer
tokenizers = TokenizerHandler(repo_id="black-forest-labs/FLUX.1-dev", max_t5_length=512, local_path=None)

clip_tokenizer = TokenizerCLIP(
    tokenizer=tokenizers.clip,
)

clip_text_encoder = CLIPTextModel(dims=768, num_encoder_layers=12)


def get_clip_embedding(text):
    tokens = clip_tokenizer.tokenizer(text, return_tensors="pt")
    # return clip_text_encoder(tokens).last_hidden_state[:, 0, :]  # Use CLS token embedding
    tokens_mlx = mx.array(tokens["input_ids"].numpy())
    return clip_text_encoder(tokens_mlx)


captions = ["A cat sitting on a mat.", "A beautiful landscape with mountains."]
embeddings = mx.stack([get_clip_embedding(c) for c in captions])  # Shape (N, D)

print(embeddings.shape)

embeddings = embeddings.squeeze(1)  # Removes singleton dim (2, 1, 768) → (2, 768)

print(embeddings.shape)


def compute_covariance(embeddings):
    N = embeddings.shape[0]
    C = (embeddings.T @ embeddings) / N  # Uncentered covariance matrix
    return C


C = compute_covariance(embeddings)

print(C.shape)

epsilon = 1e-5  # Regularization to ensure positive definiteness
C_inv = mx.linalg.inv(C + epsilon * mx.eye(C.shape[0]), stream=mx.cpu)


## Done testing now lets calculate for lion dataset

dataset = download_or_get_cached_dataset()

laion_texts = dataset["train"]["caption"]
laion_texts = laion_texts[:100000]

# Define output directory for caching
output_dir = "clip_embeddings_cache"
os.makedirs(output_dir, exist_ok=True)

batch_size = 64
num_batches = len(laion_texts) // batch_size

all_embeddings = []


def calculate_embeddings_in_batches():
    for i in tqdm(range(num_batches), desc="Processing CLIP embeddings"):
        batch_texts = laion_texts[i * batch_size : (i + 1) * batch_size]

        embeddings = mx.stack([get_clip_embedding(c) for c in batch_texts])  # Shape (N, D)
        print(embeddings.shape)

        # all_embeddings.append(embeddings)

        # # Free MLX memory
        # mx.metal.clear_cache()
        # gc.collect()

        np.save(os.path.join(output_dir, f"embeddings_batch_{i}.npy"), embeddings)


# calculate_embeddings_in_batches()
print(all_embeddings)

# Load all saved embeddings
all_embeddings = []
for i in range(num_batches):
    file_path = os.path.join(output_dir, f"embeddings_batch_{i}.npy")
    batch_embeddings = np.load(file_path)
    batch_embeddings_mlx = mx.array(batch_embeddings)

    # Squeeze extra dimension if needed (e.g., (batch, 1, 768) -> (batch, 768))
    if batch_embeddings_mlx.shape[1] == 1:
        batch_embeddings_mlx = batch_embeddings_mlx.squeeze(1)

    all_embeddings.append(batch_embeddings_mlx)
# # Concatenate all batches into a final array

X = mx.concatenate(all_embeddings, axis=0)
print(X.shape)

C = compute_covariance(X)

print(C.shape)

epsilon = 1e-5  # Regularization to ensure positive definiteness
C_inv = mx.linalg.inv(C + epsilon * mx.eye(C.shape[0]), stream=mx.cpu)


print("Covariance Matrix Shape:", C.shape)
# Expected: (768, 768) for CLIP ViT-L/14

symmetry_check = np.allclose(C, C.T, atol=1e-6)
print("Is Covariance Matrix Symmetric?", symmetry_check)  # Should be True

eigenvalues = np.linalg.eigvalsh(C)  # Compute eigenvalues
print("Min Eigenvalue:", np.min(eigenvalues))  # Should be >= 0

v = np.random.randn(C.shape[0], 1)  # Random vector
quad_form = v.T @ C @ v  # Should be >= 0
print("Quadratic Form:", quad_form.item())  # Should be non-negative


np.save("covariance_matrix.npy", C)
np.save("covariance_matrix_inv.npy", C_inv)

print("✅ Covariance matrix computed and saved successfully!")
