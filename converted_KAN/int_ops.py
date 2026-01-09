import torch


INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def _clamp_int32(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, INT32_MIN, INT32_MAX)


def quantize_to_int(x: torch.Tensor, frac_bits: int = 8) -> torch.Tensor:
    """
    Fixed-point quantization: q = round(x * 2^frac_bits).
    """
    if not torch.is_floating_point(x):
        return x.to(torch.int32)
    scale = 2**frac_bits
    q = torch.round(x * scale).to(torch.int64)
    return _clamp_int32(q).to(torch.int32)


def dequantize_from_int(q: torch.Tensor, frac_bits: int = 8) -> torch.Tensor:
    """
    Fixed-point dequantization: x = q / 2^frac_bits.
    """
    scale = 2**frac_bits
    return q.to(torch.float32) / scale


def kan_multiply_int(a: torch.Tensor, b: torch.Tensor, frac_bits: int = 8) -> torch.Tensor:
    """
    Integer-only KAN multiply using fixed-point math.
    Assumes a and b share the same fixed-point scale (2^-frac_bits).
    """
    a64 = a.to(torch.int64)
    b64 = b.to(torch.int64)

    sum_ab = a64 + b64
    diff_ab = a64 - b64
    sum_sq = sum_ab * sum_ab
    diff_sq = diff_ab * diff_ab
    diff_of_squares = sum_sq - diff_sq

    prod = torch.bitwise_right_shift(diff_of_squares, 2)
    if frac_bits > 0:
        prod = torch.bitwise_right_shift(prod, frac_bits)
    return _clamp_int32(prod).to(torch.int32)
