from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/page', methods=['GET'])
def get_page():
    return ("<p>This is the page</p>")

if __name__ == '__main__':
    app.run(debug=True) 