import os
import pathlib
from flask import Flask, request, jsonify, send_file
from privacy_score import image_analy, remove_exif, faceblur_byte
import tempfile

app= Flask(__name__)
UPLOAD_FOLDER= tempfile.gettempdir()
app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER

@app.route('/')
def index():
    uipath= pathlib.Path(__file__).parent / 'ui.html'
    if not uipath.exists():
        return "UI file not found. Use POST /image_analy with an image.", 500
    return send_file(uipath)

@app.route('/image_analy', methods=['POST'])
def image_analy():
    if 'image' not in request.files:
        return jsonify({'error': 'no image uploaded under key "image"'}), 400
    file= request.files['image']
    raw= file.read()
    try:
        result= image_analy(raw, filename_hint=file.filename)
    except Exception as e:
        return jsonify({'error': 'analysis failed', 'detail': str(e)}), 500

    #exif removed
    sani= remove_exif(raw)
    spath= os.path.join(app.config['UPLOAD_FOLDER'], f"sanitized_{file.filename}")
    with open(spath, 'wb') as f:
        f.write(sani)

    #face blur
    blur= faceblur_byte(raw)
    bpath= os.path.join(app.config['UPLOAD_FOLDER'], f"blurred_{file.filename}")
    with open(bpath, 'wb') as f:
        f.write(blur)

    result['spath']= spath
    result['bpath']= bpath
    return jsonify(result)

@app.route('/download/<kind>/<filename>')
def download(kind, filename):
    path= os.path.join(app.config['UPLOAD_FOLDER'], f"{kind}_{filename}")
    if not os.path.exists(path):
        return jsonify({'error': 'file not found'}), 404
    return send_file(path, mimetype='image/jpeg')

if __name__=='__main__':
    app.run(debug=True, port=5000)
