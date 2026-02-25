import numpy as np
from sympy import nsimplify, latex, simplify, Rational, I, sqrt, gcd, lcm, factor
from sympy import Matrix as SMatrix
import sympy
from fractions import Fraction

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



def _value_to_latex(v: float, tolerance: float = 1e-6, max_denom: int = 100) -> str:
    """
    Convert a real float to its most compact LaTeX representation.

    Priority order:
      1. Exact integer
      2. Simple fraction  p/q  with |q| <= max_denom
         Uses Python's Fraction.limit_denominator for exact rational arithmetic —
         no sympy guessing required.
      3. Square-root form — work in "squared space" (analogous to log-space × 2):
         If v^2 is a nice rational with |q| <= max_denom, then v = ±sqrt(p/q).
         Sympy then simplifies/rationalizes: e.g. sqrt(1/2) → 1/√2, sqrt(3)/3, etc.
         This handles 1/√2, 1/√3, √(2/5), 2/√3, √5/4 ... all automatically,
         replacing every old hardcoded case.
      4. Decimal fallback  (:.6g — compact, strips trailing zeros, sci-notation
         for extreme values). No ugly large integers in the output.

    Parameters
    ----------
    v         : real float to format (pass real and imaginary parts separately)
    tolerance : closeness threshold
    max_denom : maximum denominator for rational / sqrt-rational checks (default 100)
    """
    if abs(v) < tolerance:
        return "0"

    sign = -1 if v < 0 else 1
    abs_v = abs(v)

    # 1. Integer
    if abs(abs_v - round(abs_v)) < tolerance:
        return str(int(round(v)))

    # 2. Simple fraction
    frac = Fraction(abs_v).limit_denominator(max_denom)
    if abs(float(frac) - abs_v) < tolerance:
        signed_num = sign * frac.numerator
        if frac.denominator == 1:
            return str(signed_num)
        return rf"\frac{{{signed_num}}}{{{frac.denominator}}}"

    # 3. Square-root form: if v^2 is rational, v = ±sqrt(p/q)
    v2 = abs_v * abs_v
    frac2 = Fraction(v2).limit_denominator(max_denom)
    if abs(float(frac2) - v2) < tolerance and frac2 > 0:
        sym = sympy.sqrt(sympy.Rational(frac2.numerator, frac2.denominator))
        sym = sympy.simplify(sym)
        if sign == -1:
            sym = -sym
        return latex(sym)

    # 4. Decimal fallback
    return f"{v:.6g}"


# display nicely
def cleandisp(qobj, format: Literal['Dirac']=None, return_str: Literal['Markdown', 'Latex']=None,
              tolerance=1e-6, precision=12, preserve_original=False, max_denom=100):
    """
    Formats a QuTiP object (or plain numpy array/scalar) for clean LaTeX display.

    Parameters
    ----------
    max_denom : int
        Maximum denominator considered for rational / sqrt-rational simplification.
        Coefficients that don't reduce within this bound fall back to decimals.
        Default 100.
    """
    
    arr = qobj.full() if isinstance(qobj, Qobj) else np.array(qobj)
    original_shape = arr.shape
    arr = np.round(arr.real, precision) + 1j * np.round(arr.imag, precision)
    
    # --- scalar inputs ---
    if arr.ndim == 0:
        val = complex(arr)
        val = round(val.real, precision) + 1j * round(val.imag, precision)
        
        re, im = val.real, val.imag
        
        if abs(im) < tolerance:                      # purely real
            sym_str = _value_to_latex(re, tolerance, max_denom)
        elif abs(re) < tolerance:                    # purely imaginary
            coeff = _value_to_latex(im, tolerance, max_denom)
            if coeff == '1':
                sym_str = r"i"
            elif coeff == '-1':
                sym_str = r"-i"
            else:
                sym_str = coeff + r" i"
        else:                                        # genuinely complex
            re_str = _value_to_latex(re, tolerance, max_denom)
            im_str = _value_to_latex(im, tolerance, max_denom)
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
        string = _state_to_dirac(arr, tolerance, precision, preserve_original, max_denom)
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
        non_zero_mask = np.abs(arr) > tolerance
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
            rational_coeffs = []
            for e in all_entries:
                coeff, _ = e.as_coeff_Mul()
                if coeff != 0:
                    rational_coeffs.append(abs(coeff))
            
            if rational_coeffs:
                from functools import reduce
                sym_factor = reduce(lambda a, b: sympy.gcd(a, b), rational_coeffs)
                
                if sym_factor != 1 and sym_factor != 0:
                    factor_val = float(sym_factor)
                    scaled_sym = sym_arr / sym_factor

                    # Use _value_to_latex for a clean, consistent factor string
                    candidate = _value_to_latex(factor_val, tolerance, max_denom)

                    # Only pull out the factor when it produces a genuinely clean prefix
                    # (skip bare decimals — they add visual noise without clarity)
                    _looks_clean = (
                        candidate not in ("1", "0")
                        and ('\\frac' in candidate or '\\sqrt' in candidate
                             or candidate.lstrip('-').isdigit())
                    )
                    if _looks_clean:
                        scalar_str = candidate
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


def _state_to_dirac(arr, tolerance=1e-6, precision=12, preserve_original=False, max_denom=100):
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
            factor_float = float(factor)
            
            # divide by factor and check if components are integers (within tolerance)
            scaled = arr / factor_float
            scaled_components = np.concatenate([
                np.abs(scaled.real[non_zero_mask]),
                np.abs(scaled.imag[non_zero_mask])
            ])
            scaled_components = scaled_components[scaled_components > tolerance]
            
            # check if it scales to integers (within tolerance)
            if np.allclose(scaled_components, np.round(scaled_components), atol=tolerance):
                # Use _value_to_latex uniformly — handles fractions, sqrt, decimals
                candidate = _value_to_latex(factor_float, tolerance, max_denom)
                if candidate not in ("1", "0"):
                    scalar_str = candidate
                arr = scaled
    
    # Build Dirac notation
    terms = []
    for i, amplitude in enumerate(arr):
        if np.abs(amplitude) < tolerance:
            continue
        
        # Convert index to binary ket
        binary = format(i, f'0{n_qubits}b')
        ket = '|' + binary + '\\rangle'
        
        # Format the amplitude with _value_to_latex on real/imag parts
        re, im = amplitude.real, amplitude.imag
        if abs(im) < tolerance:
            coeff = _value_to_latex(re, tolerance, max_denom)
        elif abs(re) < tolerance:
            c = _value_to_latex(im, tolerance, max_denom)
            coeff = "i" if c == "1" else ("-i" if c == "-1" else f"{c} i")
        else:
            re_s = _value_to_latex(re, tolerance, max_denom)
            im_s = _value_to_latex(im, tolerance, max_denom)
            coeff = f"{re_s} - {im_s[1:]} i" if im_s.startswith('-') else f"{re_s} + {im_s} i"
        
        # Combine coefficient and ket
        if coeff == "1":
            terms.append(ket)
        elif coeff == "-1":
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




