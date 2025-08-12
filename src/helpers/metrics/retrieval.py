import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize


def top_k_retrieval(text_embeddings, image_embeddings, k):
    """
    Calculate Recall@K for text-to-image retrieval.
    
    Args:
        text_embeddings (torch.Tensor): Embeddings of texts (size: [num_texts, embedding_dim]).
        image_embeddings (torch.Tensor): Embeddings of images (size: [num_images, embedding_dim]).
        k (int): Top-k value to compute recall.
    
    Returns:
        float: Recall@k score.
    """
    
    # Normalize the embeddings to unit vectors (optional but helps improve performance)
    text_embeddings = normalize(text_embeddings.cpu().numpy(), axis=1)
    image_embeddings = normalize(image_embeddings.cpu().numpy(), axis=1)
    
    text_embeddings = torch.tensor(text_embeddings).to(text_embeddings.device)
    image_embeddings = torch.tensor(image_embeddings).to(image_embeddings.device)

    # Compute cosine similarity between text and image embeddings (text-to-image similarity)
    similarities = F.cosine_similarity(text_embeddings.unsqueeze(1), image_embeddings.unsqueeze(0), dim=-1)
    
    # Sort similarities in descending order for each text (row-wise)
    top_k_indices = similarities.argsort(dim=-1, descending=True)[:, :k]  # Top k indices
    
    # For each text, check if the corresponding image is in the top k matches
    correct = (top_k_indices == torch.arange(len(text_embeddings)).unsqueeze(1).to(text_embeddings.device)).sum()
    
    # Recall@k is the proportion of correct matches in the top k
    recall = correct.float() / len(text_embeddings)
    
    return recall.item()