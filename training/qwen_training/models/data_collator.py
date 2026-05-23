from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch

@dataclass
class SimplePadCollator:
    pad_token_id: int
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # features: input_ids (list[int]), attention_mask (list[int]), labels (list[int])
        max_len = max(len(f["input_ids"]) for f in features)

        if self.pad_to_multiple_of is not None:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            ids = f["input_ids"]
            attn = f.get("attention_mask", [1] * len(ids))
            lab = f["labels"]

            pad_len = max_len - len(ids)

            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(attn + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }