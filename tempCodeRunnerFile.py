from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('static/uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
