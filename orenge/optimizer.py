"""
Grand unified optimizer!!!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Method Selection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Smooth, unconstrained, small to medium: 
    - 'BFGS'  : general-purpose quasi-Newton, good default for smooth problems without constraints
    - 'L-BFGS-B'  : like BFGS but with box constraints, good for large problems with simple bounds
    - 'Newton'  : uses second derivatives, fast convergence near optimum but requires Hessian (or finite-diff)
        
Smooth, unconstrained, large-scale: 
    - 'CG'  : Conjugate Gradient, good for large problems where Hessian is too expensive
    - 'L-BFGS-B'  : Memory-efficient quasi-Newton, good for large problems with simple bounds
    - 'Adam'  : Popular ML optimizer, can work well for noisy gradients and large problems
        
Noisy or non-differentiable objective: 
    - 'Nelder-Mead'  : Derivative-free method for noisy/non-smooth functions
    - 'Powell'  : Conjugate direction search, no gradients needed
    - 'COBYLA'  : Constrained derivative-free optimization
    - 'DE'  : Differential Evolution, robust global optimizer for noisy functions
        
Box-constrained (simple bounds only): 
    - 'L-BFGS-B'  : Quasi-Newton with box constraints, good for smooth problems with simple bounds
    - 'TNC'  : Truncated Newton with box constraints
    - 'DA'  : Dual Annealing, global optimization with support for bounds
    - 'DE'  : Differential Evolution, robust global optimizer that can handle bounds
        
General constrained (equality + inequality): 
    - 'SLSQP'  : Sequential Quadratic Programming, good for smooth problems with general constraints
    - 'trust-constr'  : Trust-region method supporting equality & inequality constraints, more robust than SLSQP
    - 'Aug-Lag'  : Augmented Lagrangian method, can handle general constraints and is robust to local minima
        
Equality-constrained only: 
    - 'Aug-Lag'  : Augmented Lagrangian method, can handle general constraints and is robust to local minima
    - 'SLSQP'  : Sequential Quadratic Programming, good for smooth problems with general constraints
    - 'trust-constr'  : Trust-region method supporting equality & inequality constraints, more robust than SLSQP
        
Nonlinear least-squares / curve fitting: 
    - 'LM'  : Levenberg-Marquardt, fast for small to medium problems, but no constraints
    - 'LS-TRF'  : Trust-Region Reflective, handles bounds and large problems better than LM
    - 'Gauss-Newton'  : Gauss-Newton method, fast for problems where residuals are small and Jacobian is well-conditioned
        
Global optimization, continuous: 
    - 'DE'  : Differential Evolution, robust global optimizer for continuous problems
    - 'DA'  : Dual Annealing, robust global optimizer with support for bounds
    - 'CMA-ES'  : Covariance Matrix Adaptation Evolution Strategy, good for continuous problems
    - 'Basin'  : Basin-hopping, robust global optimizer for continuous problems
    - 'SHGO'  : Simultaneous Hyper-Parameter Optimization, good for continuous problems with bounds
        
Global optimization, discrete/combinatorial: 
    - 'GA'  : Genetic Algorithm, good for combinatorial problems and discrete search spaces
    - 'SA'  : Simulated Annealing, can be adapted for combinatorial problems
    - 'ACO'  : Ant Colony Optimization, good for combinatorial problems like TSP
        
Expensive black-box (few evaluations allowed): 
    - 'Bayesian'  : Bayesian Optimization, good for expensive black-box functions
    - 'TPE'  : Tree-Parzen Estimator, good for expensive black-box functions
        
Cheap black-box (many evaluations fine): 
    - 'DE'  : Differential Evolution, robust global optimizer for continuous problems
    - 'CMA-ES'  : Covariance Matrix Adaptation Evolution Strategy, good for continuous problems
    - 'PSO'  : Particle Swarm Optimization, good for continuous problems
    - 'Nelder-Mead'  : Derivative-free method for noisy/non-smooth functions
        
Multimodal (many local minima): 
    - 'DE'  : Differential Evolution, robust global optimizer for continuous problems
    - 'CMA-ES'  : Covariance Matrix Adaptation Evolution Strategy, good for continuous problems
    - 'DA'  : Dual Annealing, robust global optimizer with support for bounds
    - 'Basin'  : Basin-hopping, robust global optimizer for continuous problems
    - 'SA'  : Simulated Annealing, good for discrete and continuous problems
        
ML/NNs: 
    - 'Adam'  : Popular ML optimizer, can work well for noisy gradients and large problems
    - 'Nadam'  : Adam + Nesterov momentum, can outperform Adam in some cases
    - 'RMSProp'  : RMS gradient normalisation, good for non-stationary objectives
    - 'AdaGrad'  : Adaptive gradient method, good for sparse data and convex problems
        
Sparse recovery / LASSO-type problems: 
    - 'Proximal Gradient'  : Proximal Gradient method for composite objectives, good for problems with a smooth + non-smooth term (e.g. L1 regularization)
        
Constrained convex optimization over a simple set: 
    - 'Frank-Wolfe'  : Conditional Gradient method, good for convex optimization over simple sets like the simplex or trace-norm ball
        
Linear programming: 
    - 'LP-Simplex'  : Simplex method for linear programming
    - 'LP-IP'  : Interior-point method for linear programming
        

Solid general-purpose defaults:
 - Global problems: DE
 - Local problems: BFGS
 - ML training: Adam

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from scipy.optimize import (
    minimize,
    differential_evolution,
    basinhopping,
    shgo,
    dual_annealing,
    least_squares,
    linprog,
    approx_fprime,
)


# helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class OptimalResult:
    """optimization result container for consistent formatting"""
    x:       np.ndarray       # Best parameter vector
    fun:     float            # Objective value at x
    success: bool             # Did the optimizer converge?
    message: str              # Convergence message
    n_iter:  int              # Iterations / generations used
    n_fev:   int              # Function evaluations
    method:  str              # Which method was used
    raw:     object = field(repr=False, default=None)  # Raw library result


def _fd_gradient(func: Callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Finite-difference gradient of func at x."""
    return approx_fprime(x, func, eps)


def _fd_hessian(func: Callable, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Finite-difference Hessian of func at x (central differences)."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (func(xpp) - func(xpm) - func(xmp) + func(xmm)) / (4 * eps**2)
    return H


def _fd_jacobian(func: Callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Finite-difference Jacobian of a vector-valued function at x."""
    f0 = np.atleast_1d(func(x))
    J  = np.zeros((len(f0), len(x)))
    for j in range(len(x)):
        xp      = x.copy(); xp[j] += eps
        J[:, j] = (np.atleast_1d(func(xp)) - f0) / eps
    return J


# main show! ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Optimizer:
    """
    Parameters
    ----------
    method : str
        See module docstring for full list
    bounds : list of (min, max), optional
        Required for global methods. Also used for box constraints in
        gradient-based methods that support them (L-BFGS-B, TNC, SLSQP, etc).
    constraints : dict or list of dicts, optional
        Scipy-style constraints for SLSQP / trust-constr.
        Also used as equality constraints in Aug-Lag.
    tol : float, optional
        Convergence tolerance.
    max_iter : int, optional
        Maximum iterations / generations / evaluations.
    lr : float, optional
        Learning rate for ML optimizers (SGD, Adam, etc). Default 1e-3.
    options : dict, optional
        Extra keyword arguments forwarded directly to the underlying solver.
    """

    method_types = {
        'Minimize': {
            'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B',
            'TNC', 'SLSQP', 'COBYLA', 'trust-constr', 'trust-ncg', 'trust-krylov',
        },
        'Least-Squares': {'LM', 'LS-TRF', 'LS-dogbox'},
        'Torch': {'SGD', 'Momentum', 'Adam', 'AdaGrad', 'AdaDelta', 'RMSProp', 'Nadam'},
        'Linear Programming': {'LP-Simplex', 'LP-IP'},
    }

    def __init__(
        self,
        method: str,
        bounds: Optional[list] = None,
        constraints: Optional[Union[dict, list]] = None,
        tol: Optional[float] = None,
        max_iter: int = 1000,
        lr: float = 1e-3,
        options: Optional[dict] = None,
    ):
        self.method = method
        self.bounds = bounds
        self.constraints = constraints or []
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr
        self.options = options or {}

    def optimize(
        self,
        func:     Callable,
        x0:       Optional[np.ndarray] = None,
        jac:      Optional[Callable]   = None,
        hess:     Optional[Callable]   = None,
        callback: Optional[Callable]   = None,
    ) -> OptimalResult:
        """
        Parameters
        ----------
        func     : callable  f(x) -> scalar  (residual vector for LS methods)
        x0       : array-like, initial guess (required for local/ML methods)
        jac      : gradient callable (optional; finite-diff used if omitted)
        hess     : Hessian callable  (optional; finite-diff used if omitted)
        callback : called after each iteration with current x

        Returns
        -------
        OptimalResult
        """
        m = self.method

        if   m in self.method_types['Minimize']: return self._run_minimize(func, x0, jac, hess, callback)
        elif m in self.method_types['Least-Squares']:       return self._run_least_squares(func, x0)
        elif m in self.method_types['Torch']:    return self._run_torch(func, x0, jac)
        elif m in self.method_types['Linear Programming']:       return self._run_lp(m)
        elif m == 'DE':                   return self._run_de(func, callback)
        elif m in ('Basin', 'BH'):        return self._run_basin(func, x0, callback)
        elif m == 'SHGO':                 return self._run_shgo(func, callback)
        elif m == 'DA':                   return self._run_da(func, x0, callback)
        elif m == 'SA':                   return self._run_sa(func, x0, callback)
        elif m == 'PSO':                  return self._run_pso(func)
        elif m == 'CMA-ES':               return self._run_cmaes(func, x0)
        elif m == 'GA':                   return self._run_ga(func)
        elif m == 'ACO':                  return self._run_aco(func)
        elif m == 'Newton':               return self._run_newton(func, x0, hess, callback)
        elif m == 'Gauss-Newton':         return self._run_gauss_newton(func, x0, callback)
        elif m == 'Bayesian':             return self._run_bayesian(func)
        elif m == 'TPE':                  return self._run_tpe(func)
        elif m == 'Prox-Grad':            return self._run_prox_grad(func, x0, jac, callback)
        elif m == 'Frank-Wolfe':          return self._run_frank_wolfe(func, x0, jac, callback)
        elif m == 'Aug-Lag':              return self._run_aug_lag(func, x0, callback)
        else:
            raise ValueError(
                f"Unknown method '{m}'. See module docstring for supported methods"
            )


    # INTERNAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def _run_minimize(self, func, x0, jac, hess, callback) -> OptimalResult:
        '''
        scipy minimize
        '''
        if x0 is None:
            raise ValueError(f"'{self.method}' requires an initial guess x0.")
        r = minimize(
            func, x0,
            method      = self.method,
            jac         = jac,
            hess        = hess,
            bounds      = self.bounds,
            constraints = self.constraints,
            tol         = self.tol,
            callback    = callback,
            options     = {'maxiter': self.max_iter, **self.options},
        )
        return OptimalResult(r.x, float(r.fun), r.success, r.message,
                           r.get('nit', -1), r.get('nfev', -1), self.method, r)

    def _run_least_squares(self, func, x0) -> OptimalResult:
        '''
        scipy least squares
        '''
        if x0 is None:
            raise ValueError("Least-squares methods require x0.")
        method_map = {'LM': 'lm', 'LS-TRF': 'trf', 'LS-dogbox': 'dogbox'}
        scipy_m    = method_map[self.method]
        kw = {'method': scipy_m, 'max_nfev': self.max_iter, **self.options}
        if scipy_m != 'lm' and self.bounds:
            kw['bounds'] = tuple(zip(*self.bounds))
        r    = least_squares(func, x0, **kw)
        cost = float(0.5 * np.dot(r.fun, r.fun))
        return OptimalResult(r.x, cost, r.success, r.message, r.njev, r.nfev, self.method, r)

    def _run_torch(self, func, x0, jac) -> OptimalResult:
        '''
        torch optims
        '''
        try:
            import torch
        except ImportError:
            raise ImportError("ML optimizers require PyTorch: pip install torch")
        if x0 is None:
            raise ValueError(f"'{self.method}' requires an initial guess x0.")

        x_t  = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        opts = self.options

        optimizer_map = {
            'SGD':      lambda: torch.optim.SGD([x_t], lr=self.lr,
                            momentum=opts.get('momentum', 0.0)),
            'Momentum': lambda: torch.optim.SGD([x_t], lr=self.lr,
                            momentum=opts.get('momentum', 0.9)),
            'Adam':     lambda: torch.optim.Adam([x_t], lr=self.lr,
                            betas=opts.get('betas', (0.9, 0.999)),
                            eps=opts.get('eps', 1e-8)),
            'AdaGrad':  lambda: torch.optim.Adagrad([x_t], lr=self.lr),
            'AdaDelta': lambda: torch.optim.Adadelta([x_t],
                            rho=opts.get('rho', 0.9), eps=opts.get('eps', 1e-6)),
            'RMSProp':  lambda: torch.optim.RMSprop([x_t], lr=self.lr,
                            alpha=opts.get('alpha', 0.99)),
            'Nadam':    lambda: torch.optim.NAdam([x_t], lr=self.lr,
                            betas=opts.get('betas', (0.9, 0.999))),
        }
        opt = optimizer_map[self.method]()
        tol = self.tol or 1e-8

        for i in range(self.max_iter):
            opt.zero_grad()
            x_np    = x_t.detach().numpy()
            g       = torch.tensor(
                          jac(x_np) if jac else _fd_gradient(func, x_np),
                          dtype=torch.float64
                      )
            x_t.grad = g
            opt.step()
            if torch.norm(g).item() < tol:
                return OptimalResult(x_t.detach().numpy(),
                                   float(func(x_t.detach().numpy())),
                                   True, 'Gradient norm below tol.', i+1, i+1, self.method)

        x_final = x_t.detach().numpy()
        return OptimalResult(x_final, float(func(x_final)), False,
                           'Max iterations reached.', self.max_iter, self.max_iter, self.method)

    def _run_newton(self, func, x0, hess, callback) -> OptimalResult:
        '''
        Newton's method, custom built with finite diff gradient and Hessian (if not provided)
        '''
        if x0 is None:
            raise ValueError("Newton's method requires x0.")
        x   = np.array(x0, dtype=float)
        tol = self.tol or 1e-8
        for i in range(self.max_iter):
            g = _fd_gradient(func, x)
            H = hess(x) if hess else _fd_hessian(func, x)
            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                dx = -g
            x = x + dx
            if callback: callback(x)
            if np.linalg.norm(g) < tol:
                return OptimalResult(x, float(func(x)), True,
                                   'Gradient norm below tol.', i+1, -1, 'Newton')
        return OptimalResult(x, float(func(x)), False,
                           'Max iterations reached.', self.max_iter, -1, 'Newton')

    def _run_gauss_newton(self, func, x0, callback) -> OptimalResult:
        """
        Gauss-Newton method for nonlinear least-squares problems, custom built with finite-diff Jacobian
        """
        if x0 is None:
            raise ValueError("Gauss-Newton requires x0.")
        x   = np.array(x0, dtype=float)
        tol = self.tol or 1e-8
        for i in range(self.max_iter):
            r   = np.atleast_1d(func(x))
            J   = _fd_jacobian(func, x)
            JtJ = J.T @ J
            Jtr = J.T @ r
            try:
                dx = np.linalg.solve(JtJ, -Jtr)
            except np.linalg.LinAlgError:
                dx = -Jtr
            x = x + dx
            if callback: callback(x)
            if np.linalg.norm(Jtr) < tol:
                return OptimalResult(x, float(0.5 * np.dot(r, r)), True,
                                   'Gradient norm below tol.', i+1, -1, 'Gauss-Newton')
        r = np.atleast_1d(func(x))
        return OptimalResult(x, float(0.5 * np.dot(r, r)), False,
                           'Max iterations reached.', self.max_iter, -1, 'Gauss-Newton')

    def _run_de(self, func, callback) -> OptimalResult:
        '''
        scipy differential evolution
        '''
        if self.bounds is None: raise ValueError("DE requires bounds.")
        r = differential_evolution(
            func, self.bounds,
            maxiter  = self.max_iter,
            tol      = self.tol or 0.01,
            callback = callback,
            **self.options
        )
        return OptimalResult(r.x, float(r.fun), r.success, r.message,
                           r.get('nit', -1), r.get('nfev', -1), 'DE', r)

    def _run_basin(self, func, x0, callback) -> OptimalResult:
        '''
        scipy basin hopping
        '''
        if x0 is None: raise ValueError("Basin-Hopping requires x0.")
        min_kw = {'method': 'L-BFGS-B', 'bounds': self.bounds}
        min_kw.update(self.options.pop('minimizer_kwargs', {}))
        r   = basinhopping(func, x0, niter=self.max_iter,
                           minimizer_kwargs=min_kw, callback=callback, **self.options)
        msg = r.message[0] if isinstance(r.message, list) else r.message
        return OptimalResult(r.x, float(r.fun), r.minimization_failures == 0,
                           msg, r.nit, r.nfev, 'Basin', r)

    def _run_shgo(self, func, callback) -> OptimalResult:
        '''
        scipy shgo
        '''
        if self.bounds is None: raise ValueError("SHGO requires bounds.")
        r = shgo(func, self.bounds,
                 constraints = self.constraints or None,
                 options     = {'maxiter': self.max_iter, **self.options})
        return OptimalResult(r.x, float(r.fun), r.success, r.message,
                           r.get('nit', -1), r.get('nfev', -1), 'SHGO', r)

    def _run_da(self, func, x0, callback) -> OptimalResult:
        '''
        scipy dual annealing
        '''
        if self.bounds is None: raise ValueError("Dual Annealing requires bounds.")
        r = dual_annealing(func, self.bounds, maxiter=self.max_iter,
                           x0=x0, callback=callback, **self.options)
        return OptimalResult(r.x, float(r.fun), r.success, r.message,
                           r.get('nit', -1), r.get('nfev', -1), 'DA', r)

    def _run_sa(self, func, x0, callback) -> OptimalResult:
        """
        custom built simulated annealing with Gaussian proposals and logarithmic cooling schedule
        """
        if x0 is None and self.bounds is None:
            raise ValueError("SA requires either x0 or bounds.")
        if x0 is None:
            x0 = np.array([np.random.uniform(lo, hi) for lo, hi in self.bounds])

        T0     = self.options.get('T0', 1.0)
        step   = self.options.get('step_size', 0.1)
        x      = np.array(x0, dtype=float)
        f      = func(x)
        x_best = x.copy()
        f_best = f
        rng    = np.random.default_rng(self.options.get('seed', None))

        for k in range(1, self.max_iter + 1):
            T     = T0 / np.log(1 + k)
            x_new = x + rng.normal(0, step, size=len(x))
            if self.bounds:
                x_new = np.clip(x_new, [b[0] for b in self.bounds],
                                        [b[1] for b in self.bounds])
            f_new = func(x_new)
            delta = f_new - f
            if delta < 0 or rng.random() < np.exp(-delta / (T + 1e-300)):
                x, f = x_new, f_new
            if f < f_best:
                x_best, f_best = x.copy(), f
            if callback: callback(x)

        return OptimalResult(x_best, float(f_best), True, 'SA completed.',
                           self.max_iter, self.max_iter, 'SA')

    def _run_aco(self, func) -> OptimalResult:
        """
        custom built aco for continuous problems
        """
        if self.bounds is None: raise ValueError("ACO requires bounds.")
        n_ants  = self.options.get('n_ants', 20)
        archive = self.options.get('archive_size', 50)
        q       = self.options.get('q', 0.5)
        xi      = self.options.get('xi', 0.85)
        ndim    = len(self.bounds)
        rng     = np.random.default_rng(self.options.get('seed', None))
        lo      = np.array([b[0] for b in self.bounds])
        hi      = np.array([b[1] for b in self.bounds])

        arch_x  = rng.uniform(lo, hi, size=(archive, ndim))
        arch_f  = np.array([func(x) for x in arch_x])
        idx     = np.argsort(arch_f)
        arch_x, arch_f = arch_x[idx], arch_f[idx]

        weights  = np.exp(-np.arange(archive)**2 / (2 * q**2 * archive**2))
        weights /= weights.sum()
        best_x, best_f = arch_x[0].copy(), arch_f[0]

        for _ in range(self.max_iter):
            for _ in range(n_ants):
                chosen = rng.choice(archive, p=weights)
                sigma  = xi * np.sum(np.abs(arch_x - arch_x[chosen]), axis=0) / (archive - 1 + 1e-12)
                x_new  = np.clip(arch_x[chosen] + rng.normal(0, sigma), lo, hi)
                f_new  = func(x_new)
                if f_new < arch_f[-1]:
                    arch_x[-1] = x_new; arch_f[-1] = f_new
                    idx = np.argsort(arch_f)
                    arch_x, arch_f = arch_x[idx], arch_f[idx]
            if arch_f[0] < best_f:
                best_x, best_f = arch_x[0].copy(), arch_f[0]

        return OptimalResult(best_x, float(best_f), True, 'ACO completed.',
                           self.max_iter, self.max_iter * n_ants, 'ACO')

    def _run_pso(self, func) -> OptimalResult:
        '''
        pyswarms particle swarm optimization
        '''
        try:
            import pyswarms as ps
        except ImportError:
            raise ImportError("PSO requires pyswarms: pip install pyswarms")
        if self.bounds is None: raise ValueError("PSO requires bounds.")
        lb, ub = zip(*self.bounds)
        opts   = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        opts.update({k: v for k, v in self.options.items() if k != 'n_particles'})
        o = ps.single.GlobalBestPSO(
            n_particles = self.options.get('n_particles', 20),
            dimensions  = len(lb),
            options     = opts,
            bounds      = (np.array(lb), np.array(ub)),
        )
        cost, pos = o.optimize(func, iters=self.max_iter, verbose=False)
        return OptimalResult(pos, float(cost), True, 'PSO completed.',
                           self.max_iter, -1, 'PSO', o)

    def _run_cmaes(self, func, x0) -> OptimalResult:
        '''
        cma evolution strategy
        '''
        try:
            import cma
        except ImportError:
            raise ImportError("CMA-ES requires cma: pip install cma")
        if x0 is None: raise ValueError("CMA-ES requires x0.")
        sigma0 = self.options.pop('sigma0', 0.5)
        copts  = cma.CMAOptions()
        copts['maxiter'] = self.max_iter
        copts['verbose'] = -9
        copts.update(self.options)
        es = cma.CMAEvolutionStrategy(x0, sigma0, copts)
        es.optimize(func)
        r = es.result
        return OptimalResult(np.array(r.xbest), float(r.fbest),
                           not es.stop().get('maxiter', False),
                           str(es.stop()), r.iterations, r.evaluations, 'CMA-ES', r)

    def _run_ga(self, func) -> OptimalResult:
        '''
        pymoo genetic algorithm for global optimization over continuous spaces
        '''
        try:
            from pymoo.algorithms.soo.nonconvex.ga import GA
            from pymoo.core.problem import Problem
            from pymoo.optimize import minimize as pymoo_min
        except ImportError:
            raise ImportError("GA requires pymoo: pip install pymoo")
        if self.bounds is None: raise ValueError("GA requires bounds.")

        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])

        class _Prob(Problem):
            def __init__(self_):
                super().__init__(n_var=len(lb), n_obj=1, xl=lb, xu=ub)
            def _evaluate(self_, x, out, *args, **kwargs):
                out['F'] = np.array([[func(xi)] for xi in x])

        algo = GA(pop_size=self.options.get('pop_size', 50))
        res  = pymoo_min(_Prob(), algo,
                         termination=('n_gen', self.max_iter), verbose=False)
        return OptimalResult(res.X, float(res.F[0]), res.success,
                           'GA completed.', self.max_iter, -1, 'GA', res)

    def _run_bayesian(self, func) -> OptimalResult:
        '''
        scikit-optimize Bayesian optimization with Gaussian Process surrogate and Expected Improvement acquisition
        '''
        try:
            from skopt import gp_minimize
        except ImportError:
            raise ImportError("Bayesian requires scikit-optimize: pip install scikit-optimize")
        if self.bounds is None: raise ValueError("Bayesian requires bounds.")
        n_calls = self.options.pop('n_calls', max(50, self.max_iter))
        r = gp_minimize(func, self.bounds, n_calls=n_calls,
                        n_initial_points=self.options.pop('n_initial_points', 10),
                        **self.options)
        return OptimalResult(np.array(r.x), float(r.fun), True, 'Bayesian completed.',
                           len(r.func_vals), len(r.func_vals), 'Bayesian', r)

    def _run_tpe(self, func) -> OptimalResult:
        '''
        optuna TPE (Tree-Parzen Estimator) for hyperparameter optimization, adapted for general optimization over continuous spaces
        '''
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("TPE requires optuna: pip install optuna")
        if self.bounds is None: raise ValueError("TPE requires bounds.")

        def objective(trial):
            x = np.array([trial.suggest_float(f'x{i}', lo, hi)
                          for i, (lo, hi) in enumerate(self.bounds)])
            return func(x)

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=self.max_iter, show_progress_bar=False)
        best = study.best_params
        x    = np.array([best[f'x{i}'] for i in range(len(self.bounds))])
        return OptimalResult(x, study.best_value, True, 'TPE completed.',
                           self.max_iter, self.max_iter, 'TPE', study)

    def _run_lp(self, method: str) -> OptimalResult:
        """
        scipy linear programming
        """
        c         = np.array(self.options.get('c'))
        A_ub      = self.options.get('A_ub', None)
        b_ub      = self.options.get('b_ub', None)
        A_eq      = self.options.get('A_eq', None)
        b_eq      = self.options.get('b_eq', None)
        lp_method = 'highs-ds' if method == 'LP-Simplex' else 'highs-ipm'
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=self.bounds, method=lp_method)
        return OptimalResult(r.x, float(r.fun), r.success, r.message,
                           r.get('nit', -1), r.get('nit', -1), method, r)

    def _run_prox_grad(self, func, x0, jac, callback) -> OptimalResult:
        """
        custom built proximal gradient
        """
        if x0 is None: raise ValueError("Prox-Grad requires x0.")
        prox  = self.options.get('prox', lambda x, t: x)
        step  = self.options.get('step', self.lr)
        tol   = self.tol or 1e-8
        x     = np.array(x0, dtype=float)
        y     = x.copy()
        t_old = 1.0

        for k in range(1, self.max_iter + 1):
            g     = jac(y) if jac else _fd_gradient(func, y)
            x_new = prox(y - step * g, step)
            t_new = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
            y     = x_new + ((t_old - 1) / t_new) * (x_new - x)
            x, t_old = x_new, t_new
            if callback: callback(x)
            if np.linalg.norm(g) < tol:
                return OptimalResult(x, float(func(x)), True,
                                   'Gradient norm below tol.', k, k, 'Prox-Grad')

        return OptimalResult(x, float(func(x)), False,
                           'Max iterations reached.', self.max_iter, self.max_iter, 'Prox-Grad')

    def _run_frank_wolfe(self, func, x0, jac, callback) -> OptimalResult:
        """
        custom built frank-wolfe (conditional gradient) method for convex optimization over simple sets
        """
        if x0 is None: raise ValueError("Frank-Wolfe requires x0.")
        if self.bounds is None: raise ValueError("Frank-Wolfe requires bounds.")
        lo  = np.array([b[0] for b in self.bounds])
        hi  = np.array([b[1] for b in self.bounds])
        x   = np.clip(np.array(x0, dtype=float), lo, hi)
        tol = self.tol or 1e-8

        for k in range(1, self.max_iter + 1):
            g   = jac(x) if jac else _fd_gradient(func, x)
            s   = np.where(g < 0, hi, lo)
            d   = s - x
            eta = 2.0 / (k + 2)
            x   = x + eta * d
            if callback: callback(x)
            gap = -g @ d
            if gap < tol:
                return OptimalResult(x, float(func(x)), True,
                                   f'FW gap {gap:.2e} below tol.', k, k, 'Frank-Wolfe')

        return OptimalResult(x, float(func(x)), False,
                           'Max iterations reached.', self.max_iter, self.max_iter, 'Frank-Wolfe')

    def _run_aug_lag(self, func, x0, callback) -> OptimalResult:
        """
        custom built augmented Lagrangian method for constrained optimization, using scipy minimize for inner unconstrained steps
        """
        if x0 is None: raise ValueError("Aug-Lag requires x0.")
        eq_cons = [c for c in self.constraints if c.get('type') == 'eq']
        if not eq_cons:
            warnings.warn("Aug-Lag: no equality constraints found; falling back to L-BFGS-B.")
            return self._run_minimize(func, x0, None, None, callback)

        rho     = self.options.get('rho', 1.0)
        lam     = {i: 0.0 for i in range(len(eq_cons))}
        x       = np.array(x0, dtype=float)
        n_outer = self.options.get('n_outer', 20)
        tol     = self.tol or 1e-8

        for outer in range(n_outer):
            def augmented(x_):
                penalty = sum(
                    lam[i] * c['fun'](x_) + 0.5 * rho * c['fun'](x_)**2
                    for i, c in enumerate(eq_cons)
                )
                return func(x_) + penalty

            res = minimize(augmented, x, method='L-BFGS-B', bounds=self.bounds,
                           options={'maxiter': self.max_iter // max(n_outer, 1)})
            x = res.x
            for i, c in enumerate(eq_cons):
                lam[i] += rho * c['fun'](x)
            viol = max(abs(c['fun'](x)) for c in eq_cons)
            if callback: callback(x)
            if viol < tol:
                return OptimalResult(x, float(func(x)), True,
                                   f'Constraint violation {viol:.2e} below tol.',
                                   outer + 1, -1, 'Aug-Lag')

        return OptimalResult(x, float(func(x)), False,
                           'Max outer iterations reached.', n_outer, -1, 'Aug-Lag')
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def __repr__(self):
        return (f"Optimizer(method = '{self.method}', "
                f"bounds = {'set' if self.bounds else 'None'}, "
                f"max_iter = {self.max_iter})")