import torch
from torch import nn


class MultiHeadCoAttention(nn.Module):
    def __init__(self, multi_dim, single_dim, num_heads):
        assert multi_dim % num_heads == 0, 'multi_dim must be divisible by num_heads'
        assert single_dim % num_heads == 0, 'single_dim must be divisible by num_heads'
        super().__init__()
        self.q_proj = nn.Linear(multi_dim, multi_dim)
        self.k_proj = nn.Linear(multi_dim, multi_dim)
        self.multi_v_proj = nn.Linear(multi_dim, multi_dim)  # D'
        self.single_v_proj = nn.Linear(single_dim, single_dim)  # D

        self.multi_out_proj = nn.Linear(multi_dim, multi_dim)  # D'
        self.single_out_proj = nn.Linear(single_dim, single_dim)  # D

        self.multi_dim = multi_dim
        self.single_dim = single_dim
        self.num_heads = num_heads

    def forward(self, query, key, multi_value, single_value):
        # q, k, multi_v: (T,B,ch,D')
        # single_v: (T,B,1,D)
        query = torch.transpose(query, 0, 1)  # (B,T,ch,D')...[32, 150, 4, 64]
        key = torch.transpose(key, 0, 1)  # (B,T,ch,D')...[32, 150, 4, 64]
        multi_value = torch.permute(multi_value, (1, 2, 0, 3))  # (B,ch,T,D')...[32, 4, 150, 64]
        single_value = torch.permute(single_value, (1, 2, 0, 3))  # (B,1,T,D)...[32, 1, 150, 256]
        ###########

        q = torch.split(self.q_proj(query), self.multi_dim // self.num_heads, dim=-1)  # seq: (B,T,ch,D'/h)
        q = torch.stack(q, dim=1)  # (B,h,T,ch,D'/h)...[32, 8, 150, 4, 8]

        k = torch.split(self.k_proj(key), self.multi_dim // self.num_heads, dim=-1)  # seq: (B,T,ch,D'/h)
        k = torch.stack(k, dim=1)  # (B,h,T,ch,D'/h)...[32, 8, 150, 4, 8]

        multi_v = torch.split(self.multi_v_proj(multi_value), self.multi_dim // self.num_heads,
                              dim=-1)  # seq: (B,ch,T,D'/h)
        multi_v = torch.stack(multi_v, dim=1)  # (B, h, ch, T, D'/h)...[32, 8, 4, 150, 8]

        single_v = torch.split(self.single_v_proj(single_value), self.single_dim // self.num_heads,
                               dim=-1)  # seq: (B,1,T,D/h)
        single_v = torch.stack(single_v, dim=1)  # seq: (B,h,1,T,D/h)...[32, 32, 1, 150, 8]

        q = q.view(*q.shape[:-2], -1)  # (B, h, T, ch*D/h)
        k = k.view(*k.shape[:-2], -1)  # (B, h, T, ch*D/h)
        normalizer = torch.sqrt(torch.Tensor([float(q.shape[-1])]).to(q.device))

        sim_mat = torch.matmul(q, torch.transpose(k, -2, -1)) / normalizer  # (B, h, T, T)
        att_mat = torch.unsqueeze(nn.functional.softmax(sim_mat, dim=-1), 2)  # (B, h, 1, T, T)

        # co-attention
        multi_result = torch.matmul(att_mat, multi_v)  # (B, h, ch, T, D'/h)
        single_result = torch.matmul(att_mat, single_v)  # (B, h, 1, T, D/h)

        multi_result = torch.permute(multi_result, (3, 0, 2, 1, 4))  # (T, B, ch, h, D'/h)
        single_result = torch.permute(single_result, (3, 0, 2, 1, 4))  # (T, B, 1, h, D/h)
        multi_result = torch.reshape(multi_result, multi_result.shape[:-2] + (-1,))  # (T, B, ch, D')
        single_result = torch.reshape(single_result, single_result.shape[:-2] + (-1,))  # (T, B, 1, D)

        multi_result = self.multi_out_proj(multi_result)
        single_result = self.single_out_proj(single_result)
        return multi_result, single_result


class CoAttention(nn.Module):
    def __init__(self, embed_dim=768, single_dim=256, multi_dim=64, n_heads=8, attn_dropout=0.,
                 init_mult=1e-2):  # , pre_norm=True):
        super().__init__()
        self.init_mult = init_mult

        self.in_single_proj = nn.Linear(embed_dim, single_dim)  # single_dim == D
        self.in_single_ln = nn.LayerNorm(single_dim)

        self.in_multi_proj = nn.Linear(embed_dim, multi_dim)  # multi_dim == D'
        self.in_multi_ln = nn.LayerNorm(multi_dim)

        self.mca = MultiHeadCoAttention(multi_dim, single_dim, n_heads)
        self.mca_multi_out_ln = nn.LayerNorm(multi_dim)
        self.mca_single_out_ln = nn.LayerNorm(single_dim)

        # default MHA input: (seq, batch, feature)
        self.cross_frame_mha = nn.MultiheadAttention(single_dim, n_heads, dropout=attn_dropout, bias=True, kdim=None,
                                                     vdim=None)
        self.mha_ln = nn.LayerNorm(single_dim)

        self.cat_proj = nn.Linear(single_dim + multi_dim, embed_dim)

        self.miso = False

    def scale_weights(self):
        self.cat_proj.bias.data *= 0.
        self.cat_proj.weight.data *= self.init_mult

    def forward(self, x):
        # x: (T,B,ch,F); (150, 32, 4, 768)
        frames, B, chans, feat_dim = x.shape

        single_x = torch.mean(x, dim=-2, keepdim=True)  # (T,B,1,F)
        single_x = self.in_single_ln(self.in_single_proj(single_x))  # (T,B,1,D)

        multi_x = self.in_multi_ln(self.in_multi_proj(x))  # (T,B,ch,D')

        # MCA
        multi_mca, single_mca = self.mca(multi_x, multi_x, multi_x, single_x)  # (T,B,ch,D'), (T,B,ch,D)
        single_x = single_x + single_mca
        multi_x = multi_x + multi_mca
        multi_x = self.mca_multi_out_ln(multi_x)  # (T,B,ch,D')
        single_x = torch.squeeze(self.mca_single_out_ln(single_x), -2)  # (T,B,D)

        # MHA
        single_mha, _ = self.cross_frame_mha(single_x, single_x, single_x, need_weights=False)  # (T, B, D)
        single_x = self.mha_ln(single_mha + single_x)

        # join representations
        single_x = single_x.unsqueeze(-2)  # (T,B,1,D)
        single_x_tile = torch.tile(single_x, (1, 1, chans, 1))  # (T,B,ch,D)
        cat_x = torch.cat([single_x_tile, multi_x], dim=-1)  # (T,B,ch,D+D')
        out = self.cat_proj(cat_x)  # (T,B,ch,F)

        return out + x


class Interaction(nn.Module):
    def __init__(self, config, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([CoAttention(embed_dim=config.d_model, single_dim=config.d_model // 2,
                                                 multi_dim=config.d_model // 4, n_heads=config.encoder_attention_heads)
                                     for _ in range(num_layers)])

    def forward(self, hidden_states, per_group_sizes):
        for layer in self.layers:
            start_index = 0
            # TODO: Add attention mask to implement this efficiently
            for group in per_group_sizes:
                if group == 0:  # when decoding one of the groups can be already finished, so we skip it
                    continue
                per_group_hidden_states = hidden_states[start_index:start_index + group]
                per_group_hidden_states_transposed = layer(per_group_hidden_states.permute(1, 0, 2).unsqueeze(1))
                hidden_states[start_index:start_index + group] = per_group_hidden_states_transposed.squeeze(1).permute(
                    1, 0, 2)
                start_index += group
        return hidden_states
