from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, k, heads = 8) -> None:
        super().__init__()
        # k = size of each input vector
        # heads = number of attention heads
        self.k = k
        self.heads = heads
        self.to_keys = nn.Linear(k, k*heads, bias=False)
        self.to_query = nn.Linear(k, k*heads, bias=False)
        self.to_value = nn.Linear(k, k*heads, bias=False)

        # This unifies the output of the different attention heads into a k vector
        self.unify = nn.Linear(k*heads, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        queries = self.to_query(x).view(b, t, h, k)
        keys = self.to_keys(x).view(b, t, h, k)
        values = self.to_value(x).view(b, t, h, k)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        queries = queries/(k**(0.25))
        keys = keys/(k**(0.25))

        # get weight matrix by dot product of queries and keys and scale down the weights
        # queries_shape: (b*h, t, k)
        # keys_shape: (b*h, t, k)
        w = torch.bmm(queries,keys.transpose(1,2))
        # shape of w (b*h, t, t)

        w = F.softmax(w, dim=2)
        # w now has row-wise normalised weights

        out = torch.bmm(w, values).view(b, h, t, k)

        out = out.transpose(1,2).contiguous().view(b, t, h*k)

        return self.unify(out)