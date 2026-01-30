import torch


def max_pooling_3d(x):
    """Standard max pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    return torch.max(
        torch.max(
            torch.max(x, dim=-1, keepdim=True)[0],
            dim=-2, keepdim=True
        )[0],
        dim=-3, keepdim=True
    )[0]


def product_pooling_3d(x):
    """Product pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    return torch.prod(
        torch.prod(
            torch.prod(x, dim=-1, keepdim=True),
            dim=-2, keepdim=True
        ),
        dim=-3, keepdim=True
    )


def weighted_product_pooling_3d(x, e_exponent, epsilon=1e-4):
    """Weighted product pooling with learnable exponents.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    e_exp = e_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 1, C, K_d, K_h, K_w)
    x_safe = torch.clamp(x + epsilon, min=epsilon)
    return torch.prod(
        torch.prod(
            torch.prod(torch.pow(x_safe, e_exp), dim=-1, keepdim=True),
            dim=-2, keepdim=True
        ),
        dim=-3, keepdim=True
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
            dim=-2, keepdim=True
        ),
        dim=-3, keepdim=True
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
            dim=-2, keepdim=True
        )[0],
        dim=-3, keepdim=True
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
            dim=-2, keepdim=True
        )[0],
        dim=-3, keepdim=True
    )[0]


def min_pooling_3d(x):
    """Standard min pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    return torch.min(
        torch.min(
            torch.min(x, dim=-1, keepdim=True)[0],
            dim=-2, keepdim=True
        )[0],
        dim=-3, keepdim=True
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
            dim=-2, keepdim=True
        )[0],
        dim=-3, keepdim=True
    )[0]


def complement_weighted_min_pooling_3d(x, e_exponent, epsilon=1e-4):
    """Complement weighted min pooling.
    Input: (B, D_out, H_out, W_out, C, K_d, K_h, K_w)
    Output: (B, D_out, H_out, W_out, C)
    """
    e_exp = e_exponent.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    comp_safe = torch.clamp(1 - x + epsilon, min=epsilon)
    return 1 - torch.min(
        torch.min(
            torch.min(torch.pow(comp_safe, e_exp), dim=-1, keepdim=True)[0],
            dim=-2, keepdim=True
        )[0],
        dim=-3, keepdim=True
    )[0]


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
            dim=-2, keepdim=True
        )[0],
        dim=-3, keepdim=True
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
            dim=-2, keepdim=True
        )[0],
        dim=-3, keepdim=True
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
            dim=-2, keepdim=True
        ),
        dim=-3, keepdim=True
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
            torch.prod(1 - torch.pow(torch.pow(x_safe, r_exp), s_exp), dim=-1, keepdim=True),
            dim=-2, keepdim=True
        ),
        dim=-3, keepdim=True
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
        1 - torch.prod(
            torch.prod(
                torch.prod(1 - torch.pow(comp_safe, r_exp), dim=-1, keepdim=True),
                dim=-2, keepdim=True
            ),
            dim=-3, keepdim=True
        ),
        k_exp
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
            dim=-2, keepdim=True
        ),
        dim=-3, keepdim=True
    )
    return torch.pow(1 - prod_term, k_exp)
