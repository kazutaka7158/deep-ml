def poly_term_derivative(c: float, x: float, n: float) -> float:
    # f(x) = c * x^n => f'(x) = c * n * x ^ (n - 1)
    return c * n * x**(n-1)

