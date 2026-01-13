from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase
from tokenizers import Tokenizer

from sentence_transformers.models import StaticEmbedding
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad


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
        gol_weight: float = 0.1,  # [NEW] Hardcoded weightage for the GOL Loss
    ) -> None:
        """
        GISTEmbedLoss with added GOL (Regularization) term.
        
        Args:
            gol_weight: Weightage for the auxiliary GOL loss term derived from the equation:
                        L_S = 1/(B(B-1)) * Sum(sim(q_i, q_j)^2) + ...
        """
        super().__init__()
        self.model = model
        self.guide = guide
        self.temperature = temperature
        self.gol_weight = gol_weight  # Store the weight
        self.similarity_fct = nn.CosineSimilarity(dim=-1)
        
        if not hasattr(model, "tokenizer") or not hasattr(guide, "tokenizer"):
            raise ValueError("Both the training model and the guiding model must have a tokenizer attribute.")
        
        # modified to allow static embeddings.
        if not isinstance(model.tokenizer, (PreTrainedTokenizerBase, Tokenizer)) or not isinstance(
            guide.tokenizer, (PreTrainedTokenizerBase, Tokenizer)
        ):
             print("tokenizer not an instance if pretrainedtokenizerbase , tokenizer from tokenizers library in use.")

        self.must_retokenize = (
            model.tokenizer.get_vocab() != guide.tokenizer.get_vocab() or guide.max_seq_length < model.max_seq_length
        )
        if self.must_retokenize:
            self.tokenizer = self.model.tokenizer

            if isinstance(self.model[0], StaticEmbedding):
                print("Note: The model is of static embedding type.")

        if margin_strategy not in ("absolute", "relative"):
            raise ValueError("margin_strategy must be 'absolute' or 'relative'.")
        self.margin_strategy = margin_strategy
        self.margin = margin
        self.contrast_anchors = contrast_anchors
        self.contrast_positives = contrast_positives
        self.gather_across_devices = gather_across_devices
        self.cross_entropy_loss = nn.CrossEntropyLoss()

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

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        with torch.no_grad():
            if self.must_retokenize:

                decoded = []
                for sentence_feature in sentence_features:
                    ids = sentence_feature["input_ids"]
                    
                    if hasattr(self.tokenizer, "batch_decode"):
                        decoded.append(self.tokenizer.batch_decode(ids, skip_special_tokens=True))
                    else:

                        if "offsets" in sentence_feature:
                            flat_ids = ids.tolist()
                            offsets = sentence_feature["offsets"].tolist()
                            batch_ids_list = []
                            for i in range(len(offsets)):
                                start = offsets[i]
                                end = offsets[i+1] if i < len(offsets) - 1 else len(flat_ids)
                                batch_ids_list.append(flat_ids[start:end])
                            decoded.append(self.tokenizer.decode_batch(batch_ids_list, skip_special_tokens=True))
                        else:
                            decoded.append(self.tokenizer.decode_batch(ids.tolist(), skip_special_tokens=True))

                
                sentence_features_guide = [self.guide.tokenize(sentences) for sentences in decoded]
                sentence_features_guide = [
                    {key: value.to(self.guide.device) for key, value in sentence_feature.items()}
                    for sentence_feature in sentence_features_guide
                ]
            else:
                sentence_features_guide = sentence_features

            guide_embeddings = [
                self.guide(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features_guide
            ]


        negative = None
        negative_guide = None

        if len(embeddings) == 2:
            anchor, positive = embeddings
            anchor_guide, positive_guide = guide_embeddings
        elif len(embeddings) == 3:
            anchor, positive, negative = embeddings
            anchor_guide, positive_guide, negative_guide = guide_embeddings
        else:
            raise ValueError(f"Expected 2 or 3 embeddings, got {len(embeddings)}")
        
        batch_size = anchor.size(0)
        offset = 0

        if self.gather_across_devices:
            positive = all_gather_with_grad(positive)
            positive_guide = all_gather_with_grad(positive_guide)
            if negative is not None:
                negative = all_gather_with_grad(negative)
                negative_guide = all_gather_with_grad(negative_guide)
            
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size


        # Term 1: Sum_{i!=j} (q_i^T q_j)^2

        gol_loss_anchor = self.compute_gol_term(anchor)
        
        # Term 2: Sum_{i!=j} (p_i^T p_j)^2

        gol_loss_positive = self.compute_gol_term(positive)
        
        total_gol_loss = gol_loss_anchor + gol_loss_positive

        
        ap_sim = self.sim_matrix(anchor, positive)
        guided_ap_sim = self.sim_matrix(anchor_guide, positive_guide)

        guided_sim = guided_ap_sim.diagonal(offset=offset).view(-1, 1)

        def mask_false_negatives(guided_sim_mat, sim_mat, positive_mask: Tensor | None = None):
            if self.margin_strategy == "absolute":
                mask = guided_sim_mat > (guided_sim - self.margin)
            elif self.margin_strategy == "relative":
                mask = guided_sim_mat > (guided_sim * (1 - self.margin))

            if positive_mask is not None:
                mask = mask & ~positive_mask
            sim_mat[mask] = -torch.inf
            return sim_mat

        positive_mask = torch.zeros_like(guided_ap_sim, dtype=torch.bool)

        for i in range(batch_size):
            if i + offset < positive_mask.size(1):
                positive_mask[i, i + offset] = True

        ap_sim = mask_false_negatives(guided_ap_sim, ap_sim, positive_mask=positive_mask)
        scores = [ap_sim]

        if self.contrast_anchors:
            aa_sim = self.sim_matrix(anchor, anchor)
            guided_aa_sim = self.sim_matrix(anchor_guide, anchor_guide)
            aa_sim = mask_false_negatives(guided_aa_sim, aa_sim)
            scores.append(aa_sim)

        if self.contrast_positives:
            pp_sim = self.sim_matrix(positive[offset : offset + batch_size], positive)
            guided_pp_sim = self.sim_matrix(positive_guide[offset : offset + batch_size], positive_guide)
            pp_sim = mask_false_negatives(guided_pp_sim, pp_sim)
            scores.append(pp_sim)

        if negative is not None:
            an_sim = self.sim_matrix(anchor, negative)
            guided_an_sim = self.sim_matrix(anchor_guide, negative_guide)
            an_sim = mask_false_negatives(guided_an_sim, an_sim)
            scores.append(an_sim)

        scores = torch.cat(scores, dim=1) / self.temperature
        range_labels = torch.arange(offset, offset + batch_size, device=anchor.device)
        
        gist_loss = self.cross_entropy_loss(scores, range_labels)

        return gist_loss + (self.gol_weight * total_gol_loss)

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "guide": self.guide,
            "temperature": self.temperature,
            "margin_strategy": self.margin_strategy,
            "margin": self.margin,
            "contrast_anchors": self.contrast_anchors,
            "contrast_positives": self.contrast_positives,
            "gather_across_devices": self.gather_across_devices,
            "gol_weight": self.gol_weight,
        }