import numpy as np
from sympy import nsimplify, latex, simplify, Rational, I, sqrt, gcd, lcm, factor
from sympy import Matrix as SMatrix
import sympy

from qutip import *
from qutip import Qobj

from typing import Literal

from IPython.display import Markdown, display, Latex, clear_output


'''
Exporting notebook to pdf:

jupyter nbconvert --to webpdf notebook.ipynb

'''
def niceprint(s, header_size=0, method: Literal['Markdown', 'Latex']='Markdown'):
    
    if method == 'Markdown':
        """
        Notes:
        - need to use \n to separate header
        - for new line within the same block, use <br>
        - MathJax examples: https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Typesetting%20Equations.html
        
        Example with header:
        niceprint(r'#### this is a header' + '\n' +
                  r'$\quad$ ' + f'here is my variable: {variable} <br>')
        
        Example with math: <-- also see example in Latex section below
        niceprint(f"$\\theta = {angle_deg:.0f}^\\circ$)
        
        Another example:
        circ_str = cleandisp(circ_state, return_str='Latex')
        niceprint("**RCP Jones vector (H $\\rightarrow$ QWP at $45^\\circ$)** <br>" +
                f"$E = {circ_str}$"
                )
        """
        display(Markdown(f'{"#"*header_size} ' + s if header_size > 0 else s))
    
    elif method == 'Latex':
        """
        Notes:
        - for new line within the same block, use \\
        - use r'''...''' to avoid escaping backslashes
        - {array}{ll} for left alignment, {array}{rl} for right alignment, or {array}{cc...} for columns
        - use & to align at specific points
        - math symbols: https://www.cmor-faculty.rice.edu/~heinken/latex/symbols.pdf
        
        Example with centering:
        
        niceprint(r''' <--- use double quotes here (just can't do it here b/c of docstrings)
        \begin{array}{ll}
            &\text{hello } |1\rangle \\
            &\quad \text{point 1: }  \\
            &\quad \text{point 2: } 
        \end{array}
        ''')
        --> keep tabs like this!
        
        Example with math:
        eigenvalues = np.linalg.eigvalsh(rho_avg)
        eigen_strs = " \\quad ".join(cleandisp(e, return_str='Latex') + ',' for e in eigenvalues)
        niceprint(f"\\text{{Eigenvalues of }}\\rho: \\quad {eigen_strs}", method='Latex')
        """
        # if the string already has a LaTeX environment, display as-is
        already_wrapped = any(s.strip().startswith(tok) for tok in [
            r'\begin', r'\[', r'$$', r'$'
        ])
        if already_wrapped:
            display(Latex(s))
        else:
            # auto-wrap so plain text + math both render cleanly
            display(Latex(f"\\begin{{equation*}}{s}\\end{{equation*}}"))

# display nicely
def cleandisp(qobj, format: Literal['Dirac']=None, return_str: Literal['Markdown', 'Latex']=None, tolerance=1e-6, precision=12, preserve_original=False):
    """Formats a QuTiP object for better display with correct LaTeX formatting."""
    
    arr = qobj.full() if isinstance(qobj, Qobj) else np.array(qobj)
    original_shape = arr.shape
    arr = np.round(arr.real, precision) + 1j * np.round(arr.imag, precision)
    
    # --- scalar inputs ---
    if arr.ndim == 0:
        val = complex(arr)
        val = round(val.real, precision) + 1j * round(val.imag, precision)
        
        def _scalar_to_latex(v, tolerance):
            if abs(v - 1/np.sqrt(2)) < tolerance:
                return r"\frac{1}{\sqrt{2}}"
            elif abs(v + 1/np.sqrt(2)) < tolerance:
                return r"-\frac{1}{\sqrt{2}}"
            elif abs(v - 1/np.sqrt(3)) < tolerance:
                return r"\frac{1}{\sqrt{3}}"
            elif abs(v + 1/np.sqrt(3)) < tolerance:
                return r"-\frac{1}{\sqrt{3}}"
            else:
                sym = simplify(nsimplify(v, rational=False, tolerance=tolerance))
                sym_str = latex(sym)
                # if sympy found something ugly, fall back to decimal
                if any(str(n).isdigit() and len(str(n)) > 4 
                    for n in sympy.preorder_traversal(sym) 
                    if isinstance(n, sympy.Integer)):
                    return latex(sympy.Float(round(v, 6)))
                return sym_str
        
        re, im = val.real, val.imag
        
        if abs(im) < tolerance:                      # purely real
            sym_str = _scalar_to_latex(re, tolerance)
        elif abs(re) < tolerance:                     # purely imaginary
            coeff = _scalar_to_latex(im, tolerance)
            if coeff == '1':
                sym_str = r"i"
            elif coeff == '-1':
                sym_str = r"-i"
            else:
                sym_str = coeff + r" i"
        else:                                         # genuinely complex
            re_str = _scalar_to_latex(re, tolerance)
            im_str = _scalar_to_latex(im, tolerance)
            if im_str.startswith('-'):
                sym_str = f"{re_str} - {im_str[1:]} i"
            else:
                sym_str = f"{re_str} + {im_str} i"
        
        latex_str = f"\\begin{{equation*}}{sym_str}\\end{{equation*}}"
        if return_str == 'Latex':
            return sym_str
        elif return_str == 'Markdown':
            return f"$${sym_str}$$"
        elif return_str is not None:
            return sym_str
        else:
            return niceprint(latex_str, method='Latex')
    # ------
    
    is_state_vector = (original_shape[1] == 1 if len(original_shape) > 1 else False) or \
                      (isinstance(qobj, Qobj) and qobj.type == 'ket')
    
    if format == 'Dirac' and is_state_vector:
        string = _state_to_dirac(arr, tolerance, precision, preserve_original)
        latex_str = f"\\begin{{equation*}}{string}\\end{{equation*}}"
        if return_str == 'Latex':
            return string          # just the inner LaTeX, no wrapper
        elif return_str == 'Markdown':
            return f"$${string}$$"
        else:
            return niceprint(latex_str, method='Latex')
    
    scalar_str = ""
    if not preserve_original:
        # find nonzero common factor (smallest absolute value)
        non_zero_mask = np.abs(arr) > tolerance # filter out numerical noise within tolerance
        components = np.concatenate([
            np.abs(arr.real[non_zero_mask]),
            np.abs(arr.imag[non_zero_mask])
        ])
        components = components[components > tolerance]
        
        if len(components) > 0:
            sym_arr = SMatrix(arr).applyfunc(lambda x: nsimplify(x, rational=False, tolerance=tolerance))
    
            # Collect all non-negligible entries (flatten real and imag parts)
            all_entries = []
            for entry in sym_arr:
                re_part = sympy.re(entry)
                im_part = sympy.im(entry)
                if abs(float(re_part)) > tolerance:
                    all_entries.append(re_part)
                if abs(float(im_part)) > tolerance:
                    all_entries.append(im_part)
            
            # Find the rational prefactor: GCD of all rational coefficients
            # by expressing each entry as coeff * irrational_part
            # Strategy: find the smallest rational number that divides all entries
            rational_coeffs = []
            for e in all_entries:
                # e.as_coeff_Mul() splits into (rational_coeff, rest)
                coeff, _ = e.as_coeff_Mul()
                if coeff != 0:
                    rational_coeffs.append(abs(coeff))
            
            if rational_coeffs:
                from functools import reduce
                sym_factor = reduce(lambda a, b: sympy.gcd(a, b), rational_coeffs)
                
                if sym_factor != 1 and sym_factor != 0:
                    factor_simplified = simplify(sym_factor)
                    scaled_sym = sym_arr / sym_factor
                    
                    # Only use this factoring if it actually simplifies the entries
                    if sym_factor != 1:
                        if abs(float(sym_factor) - 1/np.sqrt(2)) < tolerance:
                            scalar_str = r"\frac{1}{\sqrt{2}}"
                        elif abs(float(sym_factor) - 1/np.sqrt(3)) < tolerance:
                            scalar_str = r"\frac{1}{\sqrt{3}}"
                        else:
                            scalar_str = latex(factor_simplified)
                        
                        sympy_matrix = scaled_sym
                        matrix_latex = latex(sympy_matrix, mat_delim='(', mat_str='array')
                        latex_str = f"\\begin{{equation*}}{scalar_str}{matrix_latex}\\end{{equation*}}"
                        markdown_str = f"$$\\begin{{array}}{{ll}}{scalar_str}{matrix_latex}\\end{{array}}$$"
                        string = f"{scalar_str}{matrix_latex}" if return_str == 'Latex' else markdown_str
                        return string if return_str is not None else niceprint(latex_str, method='Latex')
    
            # fallback: no clean factor found, display as-is
            sympy_matrix = sym_arr
    matrix_latex = latex(sympy_matrix, mat_delim='(', mat_str='array')
    latex_str = f"\\begin{{equation*}}{scalar_str}{matrix_latex}\\end{{equation*}}"
    markdown_str = f"$$\\begin{{array}}{{ll}}{scalar_str}{matrix_latex}\\end{{array}}$$"
    string = f"{scalar_str}{matrix_latex}" if return_str == 'Latex' else markdown_str

    return string if return_str is not None else niceprint(latex_str, method='Latex')

def _state_to_dirac(arr, tolerance=1e-6, precision=12, preserve_original=False):
    """Convert a state vector to Dirac notation."""
    # Flatten to 1D array
    arr = arr.flatten()
    n_states = len(arr)
    
    # Determine number of qubits
    n_qubits = int(np.log2(n_states))
    if 2**n_qubits != n_states:
        raise ValueError(f"State vector length {n_states} is not a power of 2")
    
    # Find global factor
    scalar_str = ""
    if not preserve_original:
        non_zero_mask = np.abs(arr) > tolerance
        components = np.concatenate([
            np.abs(arr.real[non_zero_mask]),
            np.abs(arr.imag[non_zero_mask])
        ])
        components = components[components > tolerance]
        
        if len(components) > 0:
            min_val = np.min(components)
            factor = nsimplify(min_val, rational=False, tolerance=tolerance)
            
            # divide by factor and check if components are integers (within tolerance)
            scaled = arr / float(factor)
            scaled_components = np.concatenate([
                np.abs(scaled.real[non_zero_mask]),
                np.abs(scaled.imag[non_zero_mask])
            ])
            scaled_components = scaled_components[scaled_components > tolerance]
            
            # check if it scales to integers (within tolerance)
            if np.allclose(scaled_components, np.round(scaled_components), atol=tolerance):
                factor_simplified = simplify(factor)
                
                # manual conversion for some forms
                if abs(float(factor) - 1/np.sqrt(2)) < tolerance:
                    scalar_str = r"\frac{1}{\sqrt{2}}"
                elif abs(float(factor) - 1/np.sqrt(3)) < tolerance:
                    scalar_str = r"\frac{1}{\sqrt{3}}"
                elif abs(float(factor) - 1/2) < tolerance:
                    scalar_str = r"\frac{1}{2}"
                elif factor_simplified != 1:
                    scalar_str = latex(factor_simplified)
                
                arr = scaled
    
    # Build Dirac notation
    terms = []
    for i, amplitude in enumerate(arr):
        if np.abs(amplitude) < tolerance:
            continue
        
        # Convert index to binary ket
        binary = format(i, f'0{n_qubits}b')
        ket = '|' + binary + '\\rangle'
        
        # Simplify amplitude
        amp_simplified = nsimplify(amplitude, rational=False, tolerance=tolerance)
        
        # Format coefficient
        if abs(amp_simplified - 1) < tolerance:
            coeff = ""
        elif abs(amp_simplified + 1) < tolerance:
            coeff = "-"
        else:
            coeff = latex(amp_simplified)
        
        # Combine coefficient and ket
        if coeff == "":
            terms.append(ket)
        elif coeff == "-":
            terms.append(f"-{ket}")
        else:
            terms.append(f"{coeff}{ket}")
    
    dirac_str = ""
    for i, term in enumerate(terms):
        if i == 0:
            dirac_str = term
        else:
            if term.startswith('-'):
                dirac_str += f" - {term[1:]}"
            else:
                dirac_str += f" + {term}"
    
    if scalar_str:
        return f"{scalar_str}\\left({dirac_str}\\right)"
    else:
        return dirac_str