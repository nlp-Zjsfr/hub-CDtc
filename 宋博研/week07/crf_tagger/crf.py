"""
线性链 CRF（手写，向量化，支持 mask 与解码期硬约束）。

记号：B = batch，L = 序列长，K = 标签数
  emissions : [B, L, K]  编码器每个位置对每个标签的打分
  mask      : [B, L]     1 = 真实 token，0 = padding（要求每行左对齐、第 0 位为 1）
  参数：
    start[K]    —— 句首转到标签 k 的分数
    trans[K,K]  —— trans[i, j] = 从标签 i 转到 j 的分数
    end[K]      —— 标签 k 转到句尾的分数

loss = log Z(x) - score(x, y)       （负对数似然，前向算法求 log Z）
decode = 维特比；hard=True 时把非法转移（O→I-x、B-x→I-y、I-x→I-y、句首→I-x）置为 -inf
"""
import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, tags):
        super().__init__()
        self.tags = list(tags)
        k = len(tags)
        self.start = nn.Parameter(torch.zeros(k))
        self.end = nn.Parameter(torch.zeros(k))
        self.trans = nn.Parameter(torch.zeros(k, k))
        nn.init.uniform_(self.start, -0.1, 0.1)
        nn.init.uniform_(self.end, -0.1, 0.1)
        nn.init.uniform_(self.trans, -0.1, 0.1)
        # 硬约束掩码：allowed[i, j] = False 表示 i→j 非法；start_allowed[j] = False 表示句首不能是 j
        allowed = torch.ones(k, k, dtype=torch.bool)
        start_allowed = torch.ones(k, dtype=torch.bool)
        for j, tj in enumerate(tags):
            if tj.startswith("I-"):
                start_allowed[j] = False
                for i, ti in enumerate(tags):
                    if ti[2:] != tj[2:]:          # 前一个不是同类型的 B/I（含 O）
                        allowed[i, j] = False
        self.register_buffer("allowed", allowed, persistent=False)
        self.register_buffer("start_allowed", start_allowed, persistent=False)

    # ── 训练：负对数似然 ──
    def forward(self, emissions, tags, mask):
        """返回 batch 平均的 NLL。tags: [B, L] long；mask: [B, L] bool"""
        mask = mask.bool()
        return (self._log_partition(emissions, mask) - self._score(emissions, tags, mask)).mean()

    def _score(self, em, tags, mask):
        B, L, _ = em.shape
        ar = torch.arange(B, device=em.device)
        score = self.start[tags[:, 0]] + em[ar, 0, tags[:, 0]]
        for t in range(1, L):
            step = self.trans[tags[:, t - 1], tags[:, t]] + em[ar, t, tags[:, t]]
            score = score + step * mask[:, t]
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)    # 每句最后一个真实标签
        return score + self.end[last]

    def _log_partition(self, em, mask):
        B, L, K = em.shape
        alpha = self.start + em[:, 0]                                     # [B, K]
        for t in range(1, L):
            # nxt[b, j] = logsumexp_i(alpha[b, i] + trans[i, j]) + em[b, t, j]
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans.unsqueeze(0), dim=1) + em[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)       # padding 处保持不变
        return torch.logsumexp(alpha + self.end, dim=1)

    # ── 推理：维特比 ──
    @torch.no_grad()
    def decode(self, emissions, mask, hard=False):
        """返回 list[list[int]]，每句只含真实长度部分。"""
        mask = mask.bool()
        B, L, K = emissions.shape
        neg = torch.finfo(emissions.dtype).min / 2
        trans, start = self.trans, self.start
        if hard:
            trans = trans.masked_fill(~self.allowed, neg)
            start = start.masked_fill(~self.start_allowed, neg)

        score = start + emissions[:, 0]                                   # [B, K]
        history = []
        for t in range(1, L):
            cand = score.unsqueeze(2) + trans.unsqueeze(0)                # [B, K(prev), K(cur)]
            best, idx = cand.max(dim=1)
            nxt = best + emissions[:, t]
            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, nxt, score)
            # padding 位置的回溯指针指向自己，回溯时等价于"原地不动"
            keep = torch.arange(K, device=emissions.device).expand(B, K)
            history.append(torch.where(m, idx, keep))
        score = score + self.end

        lengths = mask.sum(1).tolist()
        best_last = score.argmax(1)
        history = torch.stack(history, 1).cpu() if history else None      # [B, L-1, K]
        best_last = best_last.cpu()
        out = []
        for b in range(B):
            path = [int(best_last[b])]
            for t in range(L - 2, -1, -1):
                path.append(int(history[b, t, path[-1]]))
            path.reverse()
            out.append(path[:lengths[b]])
        return out
