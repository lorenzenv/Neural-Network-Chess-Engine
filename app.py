from flask import Flask, render_template, jsonify
from chess_engine_v3 import EngineV3, ENGINE_VERSION, ENGINE_NAME, ENGINE_FEATURES

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/version')
def get_version():
    """Return engine version information"""
    return jsonify({
        "version": ENGINE_VERSION,
        "name": ENGINE_NAME,
        "features": ENGINE_FEATURES
    })

# got the idea from https://github.com/brokenloop/FlaskChess/blob/master/flask_app.py
@app.route('/move/<int:depth>/<path:fen>')
def get_move(depth, fen):
    engine = EngineV3(fen)
    prediction = engine.get_move()
    return prediction

if __name__ == '__main__':
    app.run(debug=True)