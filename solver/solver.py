import logging
import sympy
from latex2sympy2 import latex2sympy


class Solver:
    def __init__(self, latex_eq: str, function: str):
        if not isinstance(latex_eq, str):
            raise TypeError("latex_eq arg should be of type string")
        if not isinstance(function, str):
            raise TypeError("function arg should be of type string")
        if not latex_eq:
            self.latex_solution = ''
            return
        self.equation = sympy.Eq(*[latex2sympy(eq_side) for eq_side in latex_eq.split('=')])
        logging.debug(self.equation)
        t = sympy.Symbol('t')

        self.solution = sympy.dsolve(self.equation.subs(function, sympy.Function(function)(t)))
        logging.debug(self.solution)
        self.latex_solution = sympy.latex(self.solution)
        logging.debug(self.latex_solution)


if __name__ == '__main__':  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG)
    solver = Solver(r'\frac{d}{dt}y=y', 'y')

