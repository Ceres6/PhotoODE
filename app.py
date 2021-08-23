from flask import Flask
from flask_socketio import SocketIO
from solver.solver import Solver

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='http://localhost:3000')


@app.route("/")
def index():
    return "Hello World!"


@app.route("/solver")
@app.route("/solver/<id>/<latex_eq>/<function>")
def solve_equation(latex_eq='', function=''):
    return {'equation': latex_eq, 'solution': Solver(latex_eq, function).latex_solution}


@socketio.on('equation')
def solve_equation(data):
    socketio.emit('solution', Solver(data['equation'], data['function']).latex_solution)


if __name__ == '__main__':
    socketio.run(app, debug=True)
