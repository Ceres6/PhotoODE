import os
from flask import Flask
from flask_socketio import SocketIO
from solver.solver import Solver

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
try:
    cors_allowed_origins = os.environ['NEXT_URL']
except KeyError:
    cors_allowed_origins = 'http://localhost:3000'

socketio = SocketIO(app, cors_allowed_origins=cors_allowed_origins)


@app.route("/")
def index():
    return "Running!"


@app.route("/solver")
@app.route("/solver/<id>/<latex_eq>/<function>")
def solve_equation(latex_eq='', function=''):
    return {'equation': latex_eq, 'solution': Solver(latex_eq, function).latex_solution}


@socketio.on('equation')
def solve_equation(data):
    socketio.emit('solution', Solver(data['equation'], data['function']).latex_solution)


if __name__ == '__main__':
    socketio.run(app, debug=True)
