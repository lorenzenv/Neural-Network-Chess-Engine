from flask import Flask, render_template, jsonify, request
from pure_nn_engine import Engine, ENGINE_VERSION, ENGINE_NAME, ENGINE_FEATURES

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
@app.route('/get_move', methods=['POST'])
def get_move():
    try:
        data = request.get_json()
        fen = data.get('fen')
        
        if not fen:
            return jsonify({'error': 'FEN position required'})
        
        print(f"🧠 Calculating move for position: {fen}")
        
        # Create pure NN engine  
        engine = Engine(fen)
        move = engine.get_move()
        
        print(f"✅ Calculated move: {move}")
        
        response = {'move': move}
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error calculating move: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)