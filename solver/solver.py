import logging
import sympy
from latex2sympy2 import latex2sympy


class Solver:
    def __init__(self, latex_eq: str, function: str):
        self.equation = sympy.Eq(*[latex2sympy(eq_side) for eq_side in latex_eq.split('=')])
        logging.debug(self.equation)
        t = sympy.Symbol('t')

        self.solution = sympy.dsolve(self.equation.subs(function, sympy.Function(function)(t)))
        logging.debug(self.solution)
        self.latex_solution = sympy.latex(self.solution)
        print(self.latex_solution)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    solver = Solver(r'\frac{d}{dt}y=y', 'y')

