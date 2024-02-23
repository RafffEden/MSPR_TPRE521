from flask import Flask, jsonify, request
import pandas as pd

app = Flask(__name__)

# Read data from CSV file
@app.route('/get_data', methods=['GET'])
def get_data():
    try:
        df = pd.read_csv('data.csv')
        return jsonify(df.to_dict(orient='records'))
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

# Post data to CSV file
@app.route('/post_data', methods=['POST'])
def post_data():
    try:
        data = request.json
        df = pd.DataFrame(data)
        df.to_csv('data.csv', mode='a', index=False, header=not bool(pd.read_csv('data.csv').shape[0]))
        return jsonify({'success': True}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
