import cmath


def fft(p, _ifft=False):
    """Apply FFT.

    Args:
        p (List[float]): Sequence.
        _ifft (bool): When True, Apply inverse FFT. For IFFT, use ifft() because the result won't be divided by N.

    Returns:
        y (List[float]): FFT-ed sequence.
    """
    n = len(p)

    if n == 1:
        return p

    ye, yo = fft(p[::2], _ifft=_ifft), fft(p[1::2], _ifft=_ifft)

    y = [0] * n
    w = 1
    if _ifft:
        wi = cmath.exp(2j * cmath.pi / n)
    else:
        wi = cmath.exp(-2j * cmath.pi / n)

    for i in range(n // 2):
        dd = w * yo[i]
        y[i] = ye[i] + dd
        y[i + n // 2] = ye[i] - dd
        w *= wi

    return y


def ifft(p):
    """Apply inverse FFT.

    Args:
        p (List[int]): Sequence.
    
    Returns:
        [something] (List(int)): IFFT-ed sequence.
    """
    return [a / len(p) for a in fft(p, _ifft=True)]


def convolve(f, g):
    """Calculate f * g.
    
    Args:
        f (List[float]): Sequence.
        g (List[float]): Sequence.
    
    Returns:
        [something] (List[float]): Convolved Sequence (length is len(f) + len(g) - 1).
    """
    len_product = len(f) + len(g) - 1

    # Round up to the smallest 2^k (>= len_product)
    n = 1 << (len_product - 1).bit_length()
    
    f += [0] * (n - len(f))
    g += [0] * (n - len(g))

    F = fft(f)
    G = fft(g)
    FG = [a * b for a, b in zip(F, G)]
    fg = ifft(FG)

    return [a.real for a in fg][:len_product]

