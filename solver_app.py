import flask
from flask_socketio import SocketIO

from solver.solver import Solver
from settings import NEXT_URL

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app, cors_allowed_origins=NEXT_URL)
print(NEXT_URL)

@app.route("/")
def index():
    return "Running!"


@app.route("/solver")
@app.route("/solver/<latex_eq>/<function>")
def solve_equation(latex_eq='', function=''):
    return {'equation': latex_eq, 'solution': Solver(latex_eq, function).latex_solution}


@socketio.on('equation')
def solve_equation(data):
    socketio.emit('solution', Solver(data['equation'], data['function']).latex_solution)


if __name__ == '__main__':
    socketio.run(app, debug=True)
