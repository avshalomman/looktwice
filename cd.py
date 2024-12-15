from transformers import LogitsProcessorList, LogitsProcessor
from typing import List
import torch
import torch.nn.functional as F

class Classifier(torch.nn.Module):
    def __init__(self, input_dim: int):
        """
        Simple linear classifier model.
        
        Args:
            input_dim: Dimension of input features
        """
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim * 2, input_dim, bias=True)
        self.linear2 = torch.nn.Linear(input_dim, input_dim, bias=True)
        self.linear3 = torch.nn.Linear(input_dim, 1, bias=True)
        self.dropout = torch.nn.Dropout(p=0.3)
    
    def forward(self, x) -> torch.Tensor:
        x = self.dropout(self.linear1(x))
        x = F.relu(x)
        x = self.dropout(self.linear2(x))
        x = F.relu(x)
        x = self.linear3(x)
        return x

# adapted from https://github.com/ZurichNLP/ContraDecode
class EnsembleLogitsProcessor(LogitsProcessor):

    def __init__(self, num_beams: int, source_weights: List[float] = [1, -0.1], preserve_bos_token: bool = False):
        self.num_beams = num_beams
        self.source_weights = source_weights
        self.preserve_bos_token = preserve_bos_token

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if self.preserve_bos_token and cur_len <= 1:
            return scores

        scores = F.softmax(scores, dim=-1)

        batch_size = int(input_ids.size(0) / self.num_beams)
        if self.source_weights is not None:
            assert len(self.source_weights) == batch_size
            source_weights = torch.Tensor(self.source_weights).to(scores.device)
        else:
            source_weights = 1/(batch_size-1) * torch.ones((batch_size,), device=scores.device)
        for i in range(self.num_beams):
            beam_indices = self.num_beams * torch.arange(batch_size, device=scores.device, dtype=torch.long) + i
            cands = scores[beam_indices]
            mean_scores = torch.log((source_weights.unsqueeze(-1).expand(-1, scores.size(-1)) * cands).sum(dim=0))
            for j in beam_indices:
                scores[j] = mean_scores

        if torch.isnan(scores).any():
            scores = torch.nan_to_num(scores, nan=float('-inf'))

        return scores