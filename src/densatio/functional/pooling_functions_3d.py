import torch


def max_pooling_3d(x):
    """Standard max pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    return torch.max(
        torch.max(torch.max(x, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0],
        dim=-3,
        keepdim=True,
    )[0]


def product_pooling_3d(x):
    """Product pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    return torch.prod(
        torch.prod(torch.prod(x, dim=-1, keepdim=True), dim=-2, keepdim=True),
        dim=-3,
        keepdim=True,
    )


def weighted_product_pooling_3d(x, e_exponent, epsilon=1e-4):
    """Weighted product pooling with learnable exponents.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    e_exp = (
        e_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    )  # (1, 1, 1, 1, C, K_d, K_h, K_w)
    x_safe = torch.clamp(x + epsilon, min=epsilon)
    return torch.prod(
        torch.prod(
            torch.prod(torch.pow(x_safe, e_exp), dim=-1, keepdim=True),
            dim=-2,
            keepdim=True,
        ),
        dim=-3,
        keepdim=True,
    )


def complement_weighted_product_pooling_3d(x, e_exponent, epsilon=1e-4):
    """Complement weighted product pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    e_exp = e_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    comp_safe = torch.clamp(1 - x + epsilon, min=epsilon)
    return 1 - torch.prod(
        torch.prod(
            torch.prod(torch.pow(comp_safe, e_exp), dim=-1, keepdim=True),
            dim=-2,
            keepdim=True,
        ),
        dim=-3,
        keepdim=True,
    )


def weighted_max_pooling_3d(x, e_exponent, epsilon=1e-4):
    """Weighted max pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    e_exp = e_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x_safe = torch.clamp(x + epsilon, min=epsilon)
    return torch.max(
        torch.max(
            torch.max(torch.pow(x_safe, e_exp), dim=-1, keepdim=True)[0],
            dim=-2,
            keepdim=True,
        )[0],
        dim=-3,
        keepdim=True,
    )[0]


def complement_weighted_max_pooling_3d(x, e_exponent, epsilon=1e-4):
    """Complement weighted max pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    e_exp = e_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    comp_safe = torch.clamp(1 - x + epsilon, min=epsilon)
    return torch.max(
        torch.max(
            torch.max(torch.pow(comp_safe, e_exp), dim=-1, keepdim=True)[0],
            dim=-2,
            keepdim=True,
        )[0],
        dim=-3,
        keepdim=True,
    )[0]


def min_pooling_3d(x):
    """Standard min pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    return torch.min(
        torch.min(torch.min(x, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0],
        dim=-3,
        keepdim=True,
    )[0]


def weighted_min_pooling_3d(x, e_exponent, epsilon=1e-4):
    """Weighted min pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """

    e_exp = e_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x_safe = torch.clamp(x + epsilon, min=epsilon)
    return torch.min(
        torch.min(
            torch.min(torch.pow(x_safe, e_exp), dim=-1, keepdim=True)[0],
            dim=-2,
            keepdim=True,
        )[0],
        dim=-3,
        keepdim=True,
    )[0]


def complement_weighted_min_pooling_3d(x, e_exponent, epsilon=1e-4):
    """Complement weighted min pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    e_exp = e_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    comp_safe = torch.clamp(1 - x + epsilon, min=epsilon)
    return (
        1
        - torch.min(
            torch.min(
                torch.min(torch.pow(comp_safe, e_exp), dim=-1, keepdim=True)[0],
                dim=-2,
                keepdim=True,
            )[0],
            dim=-3,
            keepdim=True,
        )[0]
    )


def pg_min_in_3d(x, e_exponent, epsilon=1e-4):
    """Power-Gated min input.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    e_exp = e_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x_safe = torch.clamp(x + epsilon, min=epsilon)
    return torch.max(
        torch.max(
            torch.max(1 - torch.pow(x_safe, e_exp), dim=-1, keepdim=True)[0],
            dim=-2,
            keepdim=True,
        )[0],
        dim=-3,
        keepdim=True,
    )[0]


def pg_max_inv_exp_3d(x, e_exponent, epsilon=1e-4):
    """Power-Gated max inverse exponent.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    e_exp = e_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x_safe = torch.clamp(x + epsilon, min=epsilon)
    return torch.max(
        torch.max(
            torch.max(torch.pow(x_safe, 1.0 / e_exp), dim=-1, keepdim=True)[0],
            dim=-2,
            keepdim=True,
        )[0],
        dim=-3,
        keepdim=True,
    )[0]


def pg_prod_in_3d(x, e_exponent, epsilon=1e-4):
    """Power-Gated product input.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    e_exp = e_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x_safe = torch.clamp(x + epsilon, min=epsilon)
    return torch.prod(
        torch.prod(
            torch.prod(1 - torch.pow(x_safe, e_exp), dim=-1, keepdim=True),
            dim=-2,
            keepdim=True,
        ),
        dim=-3,
        keepdim=True,
    )


def pg_prod_in_out_3d(x, r_exponent, s_exponent, epsilon=1e-4):
    """Power-Gated product input-output.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    r_exp = r_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    s_exp = s_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x_safe = torch.clamp(x + epsilon, min=epsilon)
    return 1 - torch.prod(
        torch.prod(
            torch.prod(
                1 - torch.pow(torch.pow(x_safe, r_exp), s_exp), dim=-1, keepdim=True
            ),
            dim=-2,
            keepdim=True,
        ),
        dim=-3,
        keepdim=True,
    )


def pg_prod_out_dist_3d(x, r_exponent, k_exponent, epsilon=1e-4):
    """Power-Gated product output distance.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    r_exp = r_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    k_exp = k_exponent.view(1, 1, 1, 1, 1, 1, 1, 1)
    comp_safe = torch.clamp(1 - x + epsilon, min=epsilon)
    return torch.pow(
        1
        - torch.prod(
            torch.prod(
                torch.prod(1 - torch.pow(comp_safe, r_exp), dim=-1, keepdim=True),
                dim=-2,
                keepdim=True,
            ),
            dim=-3,
            keepdim=True,
        ),
        k_exp,
    )


def pg_prod_in_out_dist_3d(x, r_exponent, s_exponent, k_exponent, epsilon=1e-4):
    """Power-Gated product input-output distance.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    r_exp = r_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    s_exp = s_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    k_exp = k_exponent.view(1, 1, 1, 1, 1, 1, 1, 1)

    x_safe = torch.clamp(x + epsilon, min=epsilon)
    inner_term = torch.clamp(1 + epsilon - torch.pow(x_safe, r_exp), min=epsilon)
    prod_term = torch.prod(
        torch.prod(
            torch.prod(torch.pow(inner_term, s_exp), dim=-1, keepdim=True),
            dim=-2,
            keepdim=True,
        ),
        dim=-3,
        keepdim=True,
    )
    return torch.pow(1 - prod_term, k_exp)


def avg_pooling_3d(x):
    """Average pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    return torch.mean(
        torch.mean(torch.mean(x, dim=-1, keepdim=True), dim=-2, keepdim=True),
        dim=-3,
        keepdim=True,
    )


def lip_pooling_3d(x, l):
    """LIP pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    weight = l.exp()
    numerator = avg_pooling_3d(x * weight)
    denominator = avg_pooling_3d(weight)
    return numerator / denominator


def soft_pooling_3d(x):
    """Soft pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    weight = x.exp()
    numerator = avg_pooling_3d(x * weight)
    denominator = avg_pooling_3d(weight)
    return numerator / denominator


def mam_pooling_3d(x):
    """MAM pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    max_val = max_pooling_3d(x)
    min_val = min_pooling_3d(x)
    avg_val = avg_pooling_3d(x)

    return torch.where(
        (max_val - min_val) > avg_val,
        max_val - min_val,
        (max_val + avg_val) / 2,
    )


def avg_topk_pooling_3d(x, k=2):
    """Average top-k pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    B, D_out, H_out, W_out, C, K_d, K_h, K_w = x.shape
    flat = x.reshape(B, D_out, H_out, W_out, C, K_d * K_h * K_w)
    k_safe = min(k, K_d * K_h * K_w)
    topk_values, _ = torch.topk(flat, k=k_safe, dim=-1)
    return topk_values.mean(dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)


def t_max_avg_pooling_3d(x, T, k=2):
    """Thresholded max-average pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    B, D_out, H_out, W_out, C, K_d, K_h, K_w = x.shape
    flat = x.reshape(B, D_out, H_out, W_out, C, K_d * K_h * K_w)
    k_safe = min(k, K_d * K_h * K_w)

    topk_values, _ = torch.topk(flat, k=k_safe, dim=-1)

    max_values = topk_values.max(dim=-1, keepdim=True)[0]
    avg_values = topk_values.mean(dim=-1, keepdim=True)

    output = torch.where(max_values >= T, max_values, avg_values)
    return output.unsqueeze(-1).unsqueeze(-1)


def lbp_pooling_3d(x):
    """Local Binary Pattern pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    B, D_out, H_out, W_out, C, K_d, K_h, K_w = x.shape

    center_d = K_d // 2
    center_h = K_h // 2
    center_w = K_w // 2

    center = x[..., center_d, center_h, center_w]

    flat = x.reshape(B, D_out, H_out, W_out, C, K_d * K_h * K_w)
    center_idx = center_d * (K_h * K_w) + center_h * K_w + center_w

    bits = (flat >= center.unsqueeze(-1)).int()

    weights = torch.zeros(K_d * K_h * K_w, dtype=torch.int64, device=x.device)
    neighbor_positions = [i for i in range(K_d * K_h * K_w) if i != center_idx]
    for bit_idx, pos in enumerate(neighbor_positions):
        weights[pos] = 1 << bit_idx

    lbp_code = (bits * weights.view(1, 1, 1, 1, 1, -1)).sum(dim=-1).float()

    n_neighbors = K_d * K_h * K_w - 1
    lbp_norm = lbp_code / (2**n_neighbors - 1)

    flat_std = flat.std(dim=-1)
    flat_max = flat.max(dim=-1).values

    out = lbp_norm * flat_std + (1 - lbp_norm) * flat_max
    return out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def lbpri_pooling_3d(x):
    """Local Binary Pattern Rotation Invariant pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    B, D_out, H_out, W_out, C, K_d, K_h, K_w = x.shape

    center_d = K_d // 2
    center_h = K_h // 2
    center_w = K_w // 2

    flat = x.reshape(B, D_out, H_out, W_out, C, K_d * K_h * K_w)
    center_idx = center_d * (K_h * K_w) + center_h * K_w + center_w

    g_c = flat[..., center_idx]

    mask = torch.ones(K_d * K_h * K_w, device=x.device, dtype=torch.bool)
    mask[center_idx] = False
    neighbors = flat[..., mask]

    n_neighbors = neighbors.size(-1)

    diffs = neighbors - g_c.unsqueeze(-1)

    abs_diffs = diffs.abs()
    D = abs_diffs.argmax(dim=-1)

    s_vals = (diffs >= 0).float()

    idx = torch.arange(n_neighbors, device=x.device).view(1, 1, 1, 1, 1, -1)
    exp = (idx - D.unsqueeze(-1) - 1) % n_neighbors
    weights = (2.0**exp).float()

    lbpri_val = (s_vals * weights).sum(dim=-1, keepdim=True)

    return lbpri_val.unsqueeze(-1).unsqueeze(-1)
