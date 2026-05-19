import torch


def max_pooling_1d(x):
    """Standard max pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    return torch.max(x, dim=-1, keepdim=True)[0]


def weighted_max_pooling_1d(x, e_exponent, epsilon=1e-8):
    """Weighted max pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    return torch.max(torch.pow(x + epsilon, e_exponent), dim=-1, keepdim=True)[0]


def complement_weighted_max_pooling_1d(x, e_exponent, epsilon=1e-2):
    """Complement weighted max pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    return torch.max(torch.pow((1 - x + epsilon), e_exponent), dim=-1, keepdim=True)[0]


def min_pooling_1d(x):
    """Standard min pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    return torch.min(x, dim=-1, keepdim=True)[0]


def weighted_min_pooling_1d(x, e_exponent, epsilon=1e-2):
    """Weighted min pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    return torch.min(torch.pow(x + epsilon, e_exponent), dim=-1, keepdim=True)[0]


def complement_weighted_min_pooling_1d(x, e_exponent, epsilon=1e-2):
    """Complement weighted min pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    x_safe = torch.clamp(x, 0.0, 1.0)
    base = torch.clamp(1 - x_safe + epsilon, min=epsilon)
    pow_term = torch.pow(base, e_exponent)
    result = 1 - torch.min(pow_term, dim=-1, keepdim=True)[0]
    return result


def pg_min_in_1d(x, e_exponent, epsilon=1e-2):
    """Power-Gated min input.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    return torch.max(1 - torch.pow(x + epsilon, e_exponent), dim=-1, keepdim=True)[0]


def pg_max_inv_exp_1d(x, e_exponent, epsilon=1e-2):
    """Power-Gated max inverse exponent.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    return torch.max(torch.pow(x + epsilon, 1.0 / e_exponent), dim=-1, keepdim=True)[0]


def pg_prod_in_1d(x, e_exponent, epsilon=1e-2):
    """Power-Gated product input.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    return torch.prod(1 - torch.pow(x + epsilon, e_exponent), dim=-1, keepdim=True)


def pg_prod_in_out_1d(x, r_exponent, s_exponent, epsilon=1e-2):
    """Power-Gated product input-output.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    return 1 - torch.prod(
        1 - torch.pow(torch.pow(x + epsilon, r_exponent), s_exponent),
        dim=-1,
        keepdim=True,
    )


def pg_prod_out_dist_1d(x, r_exponent, k_exponent, epsilon=1e-4):
    """Power-Gated product output distance.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    x_safe = torch.clamp(x, 0.0, 1.0)

    inner_base = torch.clamp(1.0 - x_safe + epsilon, min=epsilon, max=1.0 + epsilon)
    r_safe = torch.clamp(r_exponent, min=0.0, max=5.0)
    S_safe = torch.clamp(k_exponent, min=0.0, max=5.0)

    inner_term = 1.0 - torch.pow(inner_base, r_safe)
    inner_term = torch.clamp(inner_term, min=epsilon, max=1.0)

    log_term = torch.log(inner_term + epsilon)
    log_prod = torch.sum(log_term, dim=-1, keepdim=True)
    prod_term = torch.exp(log_prod)

    result = torch.pow(1.0 - prod_term, S_safe)
    result = torch.clamp(result, min=0.0, max=1.0)

    return result


def pg_prod_in_out_dist_1d(x, r_exponent, s_exponent, k_exponent, epsilon=1e-2):
    """Power-Gated product input-output distance.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    return torch.pow(
        1
        - torch.prod(
            torch.pow(1 - torch.pow(x + epsilon, r_exponent), s_exponent),
            dim=-1,
            keepdim=True,
        ),
        k_exponent,
    )


def avg_pooling_1d(x):
    """Average pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    return torch.mean(x, dim=-1, keepdim=True)


def lip_pooling_1d(x, l):
    """LIP pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    weight = l.exp()
    return avg_pooling_1d(x * weight) / avg_pooling_1d(weight)


def soft_pooling_1d(x):
    """Soft pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    weight = x.exp()
    return avg_pooling_1d(x * weight) / avg_pooling_1d(weight)


def mam_pooling_1d(x):
    """MAM pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    max_val = max_pooling_1d(x)
    min_val = min_pooling_1d(x)
    avg_val = avg_pooling_1d(x)

    return torch.where(
        (max_val - min_val) > avg_val, max_val - min_val, (max_val + avg_val) / 2
    )


def lbp_pooling_1d(x):
    """Local Binary Pattern pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    B, P, C, L = x.shape

    center_idx = L // 2
    center = x[..., center_idx]

    bits = (x >= center.unsqueeze(-1)).int()

    weights = torch.zeros(L, dtype=torch.int, device=x.device)
    neighbor_positions = [i for i in range(L) if i != center_idx]
    for bit_idx, pos in enumerate(neighbor_positions):
        weights[pos] = 1 << bit_idx

    lbp_code = (bits * weights.view(1, 1, 1, -1)).sum(dim=-1).float()

    n_neighbors = L - 1
    lbp_norm = lbp_code / (2**n_neighbors - 1)

    std_val = x.std(dim=-1)
    max_val = x.max(dim=-1).values

    out = lbp_norm * std_val + (1 - lbp_norm) * max_val
    return out.unsqueeze(-1)


def lbpri_pooling_1d(x):
    """Local Binary Pattern Rotation Invariant pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    B, P, C, L = x.shape

    center_idx = L // 2
    g_c = x[..., center_idx]

    mask = torch.ones(L, device=x.device, dtype=torch.bool)
    mask[center_idx] = False
    neighbors = x[..., mask]

    n_neighbors = neighbors.size(-1)

    diffs = neighbors - g_c.unsqueeze(-1)

    abs_diffs = diffs.abs()
    D = abs_diffs.argmax(dim=-1)

    s_vals = (diffs >= 0).float()

    idx = torch.arange(n_neighbors, device=x.device).view(1, 1, 1, -1)
    exp = (idx - D.unsqueeze(-1) - 1) % n_neighbors
    weights = (2.0**exp).float()

    lbpri_val = (s_vals * weights).sum(dim=-1, keepdim=True)

    return lbpri_val


def avg_topk_pooling_1d(x, k=2):
    """Average top-k pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    B, P, C, L = x.shape
    k_safe = min(k, L)
    topk_values, _ = torch.topk(x, k=k_safe, dim=-1)
    return topk_values.mean(dim=-1, keepdim=True)


def t_max_avg_pooling_1d(x, T, k=2):
    """Thresholded max-average pooling.
    Input: (B, P, C, L)
    Output: (B, P, C, 1)
    """
    B, P, C, L = x.shape
    k_safe = min(k, L)

    topk_values, _ = torch.topk(x, k=k_safe, dim=-1)

    max_values = topk_values.max(dim=-1, keepdim=True)[0]
    avg_values = topk_values.mean(dim=-1, keepdim=True)

    output = torch.where(max_values >= T, max_values, avg_values)

    return output