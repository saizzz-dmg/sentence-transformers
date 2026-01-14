import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer

# Assuming GISTEmbedLoss is available in your environment
from sentence_transformers.losses import GISTEmbedLoss 

class GISTWithGOLLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, guide_model: SentenceTransformer, gol_weight: float = 1.0, **gist_kwargs):
        super().__init__()
        self.model = model
        self.gol_weight = gol_weight
        
        # 1. Instantiate the original loss normally
        # We pass all extra args to the original loss (temperature, etc.)
        self.gist_loss = GISTEmbedLoss(model, guide_model, **gist_kwargs)
        
        # Storage for the embeddings "stolen" from the forward pass
        self._captured_embeddings = None

    def _hook_fn(self, module, input, output):
        """
        This function runs automatically when the model's pooling layer finishes.
        Output is typically a dictionary containing 'sentence_embedding'.
        """
        print(output) 
        print(type(output))
        if isinstance(output, dict) and 'sentence_embedding' in output:
            self._captured_embeddings = output['sentence_embedding']
        else:
            # Fallback if output is raw tensor
            self._captured_embeddings = output

    def forward(self, sentence_features: list[dict[str, Tensor]], labels: Tensor):
        # 2. Register a hook on the pooling layer (or the last layer of the model)
        # Assuming the model is a SentenceTransformer, it has a sequence of modules.
        # We usually want the output of the last module (Pooling).
        last_module = self.model[-1]
        hook_handle = last_module.register_forward_hook(self._hook_fn)

        try:
            # 3. Call the Original GIST Loss
            # This triggers the model.forward() internally.
            # Our hook will activate and save the embeddings to self._captured_embeddings
            loss_gist = self.gist_loss(sentence_features, labels)

            # 4. Retrieve captured embeddings
            embeddings = self._captured_embeddings
            
            # Sanity check to ensure hook worked
            if embeddings is None:
                raise RuntimeError("Failed to capture embeddings via hook. Check model architecture.")

            # 5. Calculate GOL (Generalized Orthogonal Loss / Your Custom Loss)
            # You now have the exact embeddings used for GIST, without a second forward pass.
            loss_gol = self.calculate_gol(embeddings, labels)

            # 6. Combine
            total_loss = loss_gist + (self.gol_weight * loss_gol)
            
            return total_loss

        finally:
            # 7. Cleanup: ALWAYS remove the hook so it doesn't persist
            hook_handle.remove()
            self._captured_embeddings = None

    def calculate_gol(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """
        Implement your GOL logic here.
        """
        # Placeholder logic
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)