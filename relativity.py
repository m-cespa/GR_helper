import sympy as sp
from sympy import Symbol
from typing import List

class GeneralRelativity:
    def __init__(self, coords: List[Symbol], g: sp.Matrix):
        self.coords = coords
        self.g = g
        self.n = len(coords)
        self.g_inv = g.inv()

    def find_christoffel_symbols(self):
        Gamma = sp.MutableDenseNDimArray.zeros(self.n, self.n, self.n)
        for rho in range(self.n):
            for mu in range(self.n):
                for nu in range(self.n):
                    term = sum(
                        self.g_inv[rho, sigma] * (
                            sp.diff(self.g[sigma, nu], self.coords[mu]) +
                            sp.diff(self.g[sigma, mu], self.coords[nu]) -
                            sp.diff(self.g[mu, nu], self.coords[sigma])
                        ) for sigma in range(self.n)
                    )
                    Gamma[rho, mu, nu] = sp.simplify(0.5 * term)
        return Gamma

    def find_riemann_tensor(self, Gamma: sp.MutableDenseNDimArray):
        Riemann = sp.MutableDenseNDimArray.zeros(self.n, self.n, self.n, self.n)
        for d in range(self.n):
            for a in range(self.n):
                for b in range(self.n):
                    for c in range(self.n):
                        term1 = sp.diff(Gamma[d, b, c], self.coords[a])
                        term2 = sp.diff(Gamma[d, a, c], self.coords[b])
                        term3 = sum(Gamma[e, a, c] * Gamma[d, b, e] for e in range(self.n))
                        term4 = sum(Gamma[e, b, c] * Gamma[d, a, e] for e in range(self.n))
                        Riemann[d, a, b, c] = sp.simplify(-term1 + term2 + term3 - term4)
        return Riemann
