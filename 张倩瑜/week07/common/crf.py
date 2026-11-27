# -*- coding: utf-8 -*-
"""轻量 CRF：负对数似然 + Viterbi。"""
import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        self.start = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end = nn.Parameter(torch.randn(num_tags) * 0.1)

    def _score_path(self, emissions, tags, mask):
        # emissions: (B,L,T) tags:(B,L) mask:(B,L)
        B, L, T = emissions.shape
        score = self.start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for i in range(1, L):
            m = mask[:, i]
            trans = self.transitions[tags[:, i - 1], tags[:, i]]
            emit = emissions[:, i].gather(1, tags[:, i:i + 1]).squeeze(1)
            score = score + (trans + emit) * m
        lengths = mask.long().sum(1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        score = score + self.end[last_tags]
        return score

    def _log_partition(self, emissions, mask):
        B, L, T = emissions.shape
        alpha = self.start + emissions[:, 0]
        for i in range(1, L):
            emit = emissions[:, i].unsqueeze(1)  # (B,1,T)
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emit
            new_alpha = torch.logsumexp(scores, dim=1)
            m = mask[:, i].unsqueeze(1)
            alpha = torch.where(m.bool(), new_alpha, alpha)
        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def neg_log_likelihood(self, emissions, tags, mask):
        gold = self._score_path(emissions, tags, mask)
        Z = self._log_partition(emissions, mask)
        return (Z - gold).mean()

    def decode(self, emissions, mask):
        B, L, T = emissions.shape
        device = emissions.device
        score = self.start + emissions[:, 0]
        back = torch.zeros(B, L, T, dtype=torch.long, device=device)
        for i in range(1, L):
            sc = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            max_score, idx = sc.max(dim=1)
            m = mask[:, i].unsqueeze(1).bool()
            score = torch.where(m, max_score, score)
            back[:, i] = idx
        score = score + self.end.unsqueeze(0)
        best_last = score.argmax(dim=1)
        paths = torch.zeros(B, L, dtype=torch.long, device=device)
        paths[:, 0] = best_last  # placeholder
        for b in range(B):
            length = int(mask[b].sum().item())
            tag = best_last[b].item()
            seq = [tag]
            for i in range(length - 1, 0, -1):
                tag = back[b, i, tag].item()
                seq.append(tag)
            seq.reverse()
            paths[b, :length] = torch.tensor(seq, device=device)
        return paths
