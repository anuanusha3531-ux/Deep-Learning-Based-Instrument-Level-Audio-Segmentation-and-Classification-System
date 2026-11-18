from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory

import os

from predict import predict_instruments


# Flask App
app = Flask(__name__)


# Upload Folder
UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Create Upload Folder
if not os.path.exists(UPLOAD_FOLDER):

    os.makedirs(UPLOAD_FOLDER)


# Route for Uploaded Audio
@app.route('/uploads/<filename>')
def uploaded_file(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


# Home Route
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Check uploaded file
        if "audio" not in request.files:

            return "No file uploaded!"

        audio_file = request.files["audio"]

        # Empty filename
        if audio_file.filename == "":

            return "No selected file!"

        # Save uploaded file
        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            audio_file.filename
        )

        audio_file.save(file_path)

        print("\nFile Uploaded Successfully!")

        print("Saved Path:", file_path)

        # Default playback file
        playback_file = audio_file.filename

        # Convert MP4 to WAV for browser playback
        if file_path.endswith(".mp4"):

            print("\nConverting MP4 to WAV...")

            wav_filename = (
                os.path.splitext(
                    audio_file.filename
                )[0]
                + ".wav"
            )

            wav_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                wav_filename
            )

            # FFmpeg conversion
            os.system(
                f'ffmpeg -y -i "{file_path}" "{wav_path}"'
            )

            print("\nConversion Completed!")

            # Use WAV for browser playback
            playback_file = wav_filename

        # Predict Instruments
        results = predict_instruments(
            file_path
        )

        # Render Result Page
        return render_template(

            "result.html",

            results=results,

            audio_filename=playback_file
        )

    # GET Request
    return render_template(
        "index.html"
    )


# Run Flask App
if __name__ == "__main__":

    app.run(
        debug=True
    )