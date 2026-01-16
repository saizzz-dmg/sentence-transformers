import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
from typing import  Literal

from sentence_transformers.losses import GISTEmbedLoss 

class GISTWithGOLLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        guide: SentenceTransformer,
        temperature: float = 0.01,
        margin_strategy: Literal["absolute", "relative"] = "absolute",
        margin: float = 0.0,
        contrast_anchors: bool = True,
        contrast_positives: bool = True,
        gather_across_devices: bool = False,
        gol_weight : float = 0.1
    ) -> None:
        """
        This loss is used to train a SentenceTransformer model using the GISTEmbed algorithm.
        It takes a model and a guide model as input, and uses the guide model to guide the
        in-batch negative sample selection. The cosine similarity is used to compute the loss
        and the temperature parameter is used to scale the cosine similarities.

        You can apply different false-negative filtering strategies to discard hard negatives that are too similar to
        the positive. Two strategies are supported:

            - "absolute": Discards negatives whose similarity score is greater than or equal to ``positive_score - margin``.
            - "relative": Discards negatives whose similarity score is greater than or equal to ``positive_score * (1 - margin)``.

        In addition to the functionalities from GistEmbed Loss , we additionally include GLobal Orthogonal Loss (GOL).
        The ultimate aim is that the vectors in high dimensional space should not be anisotropic in nature and be spread across 
        all the available directions. This solved the problem of vector clouding which results in better performance in ANN algos.
        The GOL is used as an auxiliary loss and is thus important to be given less weightage. 

        This makes sure the vectors instead being accidentilly closer , will now be solely based on semantics.

        Args:
            model: SentenceTransformer model based on a `transformers` model or 'StaticEmbedding'.
            guide: SentenceTransformer model to guide the in-batch negative sample selection.
            temperature: Temperature parameter to scale the cosine similarities. Inverse of the ``scale`` parameter
                in :class:`MultipleNegativesRankingLoss`.
            margin_strategy: Strategy used for false negative filtering. One of {"absolute", "relative"}.
            margin: The margin value for filtering negatives. Defaults to 0.0, together with the "absolute" strategy,
                this only removes negatives that are more similar to the query than the positive is to the query.
            contrast_anchors: If True, include anchor-anchor pairs in the loss computation, resulting in the embeddings
                of the anchors being pushed further apart. Defaults to True, following the original GISTEmbed paper.
            contrast_positives: If True, include positive-positive pairs in the loss computation, resulting in the embeddings
                of the positives being pushed further apart. Defaults to True, following the original GISTEmbed paper,
                but setting to False may yield better results in some retrieval tasks.
            gather_across_devices: If True, gather the embeddings across all devices before computing the loss.
                Recommended when training on multiple GPUs, as it allows for larger batch sizes, but it may slow down
                training due to communication overhead, and can potentially lead to out-of-memory errors.
            gol_weight : Weight to be given to GOL. Usually low.

        References:
            - For further details, see: https://huggingface.co/papers/2402.16829

        Requirements:
            1. (anchor, positive, negative) triplets
            2. (anchor, positive) pairs

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - :class:`MultipleNegativesRankingLoss` is similar to this loss, but it does not use
              a guide model to guide the in-batch negative sample selection. `GISTEmbedLoss` yields
              a stronger training signal at the cost of some training overhead.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                guide = SentenceTransformer("all-MiniLM-L6-v2")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.GISTWithGOLLoss(model, guide)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """

        super().__init__()
        self.model = model
        self.gol_weight = gol_weight
        self.similarity_fct = nn.CosineSimilarity(dim=-1)
        
        #initializing the original loss with the arguments we got for the custom loss.

        self.gist_loss = GISTEmbedLoss(
            model = model,
            guide = guide,
            temperature=temperature , 
            margin_strategy= margin_strategy , 
            margin = margin ,
            contrast_anchors= contrast_anchors , 
            contrast_positives= contrast_positives , 
            gather_across_devices= gather_across_devices)
        

    def _hook_fn(self, module, input, output):
        """
        This function runs automatically when the model's pooling layer finishes.
        Output is typically a dictionary containing 'sentence_embedding'.
        """
        if isinstance(output, dict) and 'sentence_embedding' in output:
            self._captured_embeddings.append(output['sentence_embedding'])


    def forward(self, sentence_features: list[dict[str, Tensor]], labels: Tensor):

        #getting the final module of the model depending on DDP use
        if hasattr(self.model, "module"):
            target_module = self.model.module 
        else:
            target_module = self.model
        
        self._captured_embeddings = []
        hook_handle = target_module.register_forward_hook(self._hook_fn)

        try:
            # run standard GistEmbed loss on the sentence_features
            loss_gist = self.gist_loss(sentence_features, labels)

            # Checking if hook has been hit
            if self._captured_embeddings:
                embeddings = self._captured_embeddings
            
            elif hasattr(self.model.forward, "cache"):

                #forward modified. cache hit
                decorator_forward = self.model.forward
                
                #utilizing the full precision vectors from cache
                full_outputs = decorator_forward.cache[:len(sentence_features)]
                
                # CRITICAL: We must manually shrink to the current dimension
                # so GOL penalizes the subspace, not the full space.
                current_dim = decorator_forward.dim
                embeddings = [fo["sentence_embedding"][..., :current_dim] for fo in full_outputs]
                embeddings = [torch.nn.functional.normalize(embedding, p=2, dim=-1) for embedding in embeddings]

                #important key changes done : 
                # This loss calls GistEmbedLoss which every time , calls a forward for the model.
                # As a result , the mrl dimensions and the indices will be exhausted by the original 
                #loss . As a result , even when we cache , this auxiliary loss will further look for 
                #dimensions which will result in index out of bounds. To not affect the original 
                #implementation from Matrayoshka loss , direct cache hit has been made to get the 
                #full precision embeddings.

            else:
                raise FileNotFoundError("Hook not trigger successfully. Cache miss occured !")

            loss_gol = self.calculate_gol(embeddings)

            return loss_gist + (self.gol_weight * loss_gol)
        
        finally:
            #hook cleanup
            hook_handle.remove()
            self._captured_embeddings = None

    def calculate_gol(self , embeddings):
        """
        Docstring for calculate_gol
        
        GOL as per reference from embedding gemma , was applied only to anchors , 
        and positives. ,thus filtering out them from the embedding list.

        """
        loss = [self.compute_gol_term(embedding) for embedding in embeddings[:2]]            
        return sum(loss)


    def sim_matrix(self, embed1: Tensor, embed2: Tensor) -> Tensor:
        return self.similarity_fct(embed1.unsqueeze(1), embed2.unsqueeze(0))

    def compute_gol_term(self, embeddings: Tensor) -> Tensor:
        """
        Calculates the regularization term: 1/(B(B-1)) * Sum_{i!=j} (sim(e_i, e_j)^2)
        """
        batch_size = embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device)
            
        # 1. Compute similarity matrix (B x B)
        sim_matrix = self.sim_matrix(embeddings, embeddings)
        
        # 2. Square the similarities: (sim)^2
        sim_sq = sim_matrix.pow(2)
        
        # 3. Mask the diagonal (we only want i != j)
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        sim_sq.masked_fill_(eye_mask, 0.0)
        
        # 4. Sum and Normalize
        # Normalization factor is B * (B - 1)
        normalization = batch_size * (batch_size - 1)
        
        return sim_sq.sum() / normalization