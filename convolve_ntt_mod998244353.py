MOD = 998244353
PRIMITIVE_ROOT = 3


def ntt(p, _intt=False):
    """Apply NTT.

    Args:
        p (List[int]): Sequence.
        _intt (bool): When True, Apply inverse NTT. For INTT, use intt() because the result won't be divided by N.

    Returns:
        y (List[int]): NTT-ed sequence.
    """
    n = len(p)

    if n == 1:
        return p

    ye, yo = ntt(p[::2], _intt=_intt), ntt(p[1::2], _intt=_intt)

    y = [0] * n
    w = 1
    if _intt:
        wi = pow(PRIMITIVE_ROOT, MOD - 2, MOD)
    else:
        wi = PRIMITIVE_ROOT

    wi = pow(wi, (MOD - 1) >> (n - 1).bit_length(), MOD)

    for i in range(n):
        dd = w * yo[i % (n // 2)]
        y[i] = (ye[i % (n // 2)] + dd) % MOD
        w *= wi
        w %= MOD

    return y


def intt(p):
    """Apply inverse NTT.

    Args:
        p (List[int]): Sequence.
    
    Returns:
        [something] (List(int)): INTT-ed sequence.
    """
    len_inv = pow(len(p), MOD - 2, MOD)
    return [a * len_inv % MOD for a in ntt(p, _intt=True)]


def convolve(f, g):
    """Calculate f * g.
    
    Args:
        f (List[int]): Sequence.
        g (List[int]): Sequence.
    
    Returns:
        [something] (List[int]): Convolved Sequence (length is len(f) + len(g) - 1).
    """
    len_product = len(f) + len(g) - 1

    # Round up to the smallest 2^k (>= len_product)
    n = 1 << (len_product - 1).bit_length()
    
    f += [0] * (n - len(f))
    g += [0] * (n - len(g))

    F = ntt(f)
    G = ntt(g)
    FG = [a * b % MOD for a, b in zip(F, G)]
    fg = intt(FG)

    return fg[:len_product]


"""
Verified on ACLPC F - Convolution
PyPy3 (3975ms): https://atcoder.jp/contests/practice2/submissions/24576298

n, m = map(int, input().split())
a = [*map(int, input().split())]
b = [*map(int, input().split())]
print(*convolve(a, b))
"""