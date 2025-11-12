# app.py
import os
import pathlib
from flask import Flask, request, jsonify, send_file
from privacy_score import analyze_image_bytes, remove_exif, blur_faces_in_bytes
import tempfile

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    ui_path = pathlib.Path(__file__).parent / 'ui.html'
    if not ui_path.exists():
        return "UI file not found. Use POST /analyze with an image.", 500
    return send_file(ui_path)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'no image uploaded under key "image"'}), 400
    file = request.files['image']
    raw = file.read()
    try:
        result = analyze_image_bytes(raw, filename_hint=file.filename)
    except Exception as e:
        return jsonify({'error': 'analysis failed', 'detail': str(e)}), 500

    # produce sanitized version with EXIF removed
    sanitized = remove_exif(raw)
    sanitized_path = os.path.join(app.config['UPLOAD_FOLDER'], f"sanitized_{file.filename}")
    with open(sanitized_path, 'wb') as f:
        f.write(sanitized)

    # also produce a face-blurred version
    blurred = blur_faces_in_bytes(raw)
    blurred_path = os.path.join(app.config['UPLOAD_FOLDER'], f"blurred_{file.filename}")
    with open(blurred_path, 'wb') as f:
        f.write(blurred)

    # attach links/paths (for demo only)
    result['sanitized_path'] = sanitized_path
    result['blurred_path'] = blurred_path
    return jsonify(result)

@app.route('/download/<kind>/<filename>')
def download(kind, filename):
    # kind = sanitized or blurred ; naive mapping
    path = os.path.join(app.config['UPLOAD_FOLDER'], f"{kind}_{filename}")
    if not os.path.exists(path):
        return jsonify({'error': 'file not found'}), 404
    return send_file(path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
