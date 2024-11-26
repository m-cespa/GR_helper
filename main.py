import streamlit as st
import sympy as sp
from sympy import latex
import numpy as np
from relativity import GeneralRelativity

st.title("General Relativity Helper")

# user input for metric dimensions
n = st.number_input("Enter the dimension of the metric (n x n):", min_value=1, max_value=10, step=1, value=2)

st.write("### Define the Coordinates:")
coords_input = st.text_input(
    "Enter coordinate names separated by spaces (e.g., `x y z`):", value=" ".join([f"x{i+1}" for i in range(n)])
)
coords = sp.symbols(coords_input)

if len(coords) != n:
    st.error(f"Please provide exactly {n} coordinate names.")
else:
    # list to fill with sublists of metric rows
    metric_entries = []

    for i in range(n):
        cols = st.columns(n)
        row_entries = []
        
        for j in range(n):
            # latex formatted title for each entry
            title = f"g_{{{i},{j}}}"
            value = cols[j].text_input(f"${title}$", value="")
            row_entries.append(value)
        
        metric_entries.append(row_entries)

    try:
        # read each string entry from the metric_entries list and convert to symbolic
        sympy_matrix = sp.Matrix([
            [sp.sympify(entry) for entry in row]for row in metric_entries
        ])

        st.write("SymPy Matrix Representation:")
        st.latex(sp.latex(sympy_matrix))

        relativity = GeneralRelativity(coords, sympy_matrix)

        # christoffel calculation
        if st.button("Calculate Christoffel Symbols"):
            st.write("### Christoffel Symbols:")
            Gamma = relativity.find_christoffel_symbols()

            for rho in range(n):
                for mu in range(n):
                    for nu in range(n):
                        if Gamma[rho, mu, nu] != 0:
                            st.latex(f"\\Gamma^{{{latex(coords[rho])}}}_{{{latex(coords[mu])} {latex(coords[nu])}}} = {latex(Gamma[rho, mu, nu])}")

            all_zero = all(sp.simplify(Gamma[rho, mu, nu]) == 0 for rho in range(n) for mu in range(n) for nu in range(n))
            if all_zero:
                st.latex(r" \text{All } \\Gamma = 0 ")

        # riemann calculation
        if st.button("Calculate Riemann Tensor"):
            st.write("### Riemann Curvature Tensor:")
            Gamma = relativity.find_christoffel_symbols()
            Riemann = relativity.find_riemann_tensor(Gamma)

            for d in range(n):
                for a in range(n):
                    for b in range(n):
                        for c in range(n):
                            if Riemann[d, a, b, c] != 0:
                                st.latex(f"R^{{{latex(coords[d])}}}_{{{latex(coords[a])} {latex(coords[b])} {latex(coords[c])}}} = {latex(Riemann[d, a, b, c])}")

            all_zero = all(sp.simplify(Riemann[d, a, b, c]) == 0 for d in range(n) for a in range(n) for b in range(n) for c in range(n))
            if all_zero:
                st.latex(r" \text{All } R = 0 ")

    except Exception as e:
        st.error(f"Error in processing the matrix or coordinates: {e}")
