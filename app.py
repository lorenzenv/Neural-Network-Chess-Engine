from flask import Flask, render_template
from chess_engine import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/move/<int:depth>/<path:fen>')
def get_move(depth, fen):
    engine = Engine(fen)
    prediction = engine.get_move()
    return prediction



if __name__ == '__main__':
    app.run(debug=True)