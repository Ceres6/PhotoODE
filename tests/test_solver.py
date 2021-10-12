import unittest

from .context import solver


class TestSolver(unittest.TestCase):
    def test_should_raise_exception_on_non_string_inputs(self):
        self.assertRaises(TypeError, solver.Solver, 1, '')
        self.assertRaises(TypeError, solver.Solver, '', 1)

    def test_should_return_empty_string_on_empty_string_input(self):
        latex_solution = solver.Solver('', '').latex_solution
        self.assertEqual(latex_solution, '')

    def test_should_return_constant_when_derivative_is_zero(self):
        latex_solution = solver.Solver(r'\frac{d}{dt}y=0', 'y').latex_solution
        self.assertEqual(latex_solution, r'y{\left(t \right)} = C_{1}')

    def test_should_return_exp_when_derivative_is_the_function(self):
        latex_solution = solver.Solver(r'\frac{d}{dt}y=y', 'y').latex_solution
        self.assertEqual(latex_solution, r'y{\left(t \right)} = C_{1} e^{t}')


if __name__ == '__main__':
    unittest.main()
