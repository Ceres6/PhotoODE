from flask import Flask
from solver.solver import Solver
app = Flask(__name__)


@app.route("/")
def index():
    return "Hello World!"


@app.route("/solver")
@app.route("/solver/<latex_eq>/<function>")
def solve_equation(latex_eq='', function=''):
    return { 'equation': latex_eq, 'solution':Solver(latex_eq, function).latex_solution}


if __name__ == '__main__':
    app.run(debug=True)
