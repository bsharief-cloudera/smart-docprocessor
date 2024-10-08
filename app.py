from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import os
import time
from werkzeug.utils import secure_filename
from docx import Document
import llm_proofer
import importlib
from python_redlines.engines import XmlPowerToolsEngine

# Reload the module containing the function
importlib.reload(llm_proofer)
from copy import deepcopy
from config import settings

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.secret_key = "supersecretkey"  # For flash messages


# Create the uploads folder if it doesn't exist
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])


# Dummy proofreading function
def dummy_proofread(input_path, edit_path, correction_path):
    # Create a document object for editing
    input_doc = Document(input_path)
    modified_doc = deepcopy(input_doc)
    # Create a document object for corrections
    corrections_doc = Document()
    modified_doc, corrections_doc = llm_proofer.proofread_and_correct_document(
        modified_doc, corrections_doc
    )
    modified_doc.save(edit_path)
    corrections_doc.save(correction_path)

    return "Processing completed successfully"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(input_path)

        # Create paths for edit and corrections documents
        edit_path = os.path.join(
            app.config["UPLOAD_FOLDER"], filename.split(".")[0] + "_edited.docx"
        )
        correction_path = os.path.join(
            app.config["UPLOAD_FOLDER"], filename.split(".")[0] + "_corrections.docx"
        )
        tracked_correction_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename.split(".")[0] + "_tracked_corrections.docx",
        )

        flash("Document has been uploaded successfully!")
        return render_template("index.html", file_uploaded=True, filename=filename)


@app.route("/proofread/<filename>", methods=["POST"])
def proofread_document(filename):
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    edit_path = os.path.join(
        app.config["UPLOAD_FOLDER"], filename.split(".")[0] + "_edited.docx"
    )
    correction_path = os.path.join(
        app.config["UPLOAD_FOLDER"], filename.split(".")[0] + "_corrections.docx"
    )
    tracked_correction_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename.split(".")[0] + "_tracked_corrections.docx",
    )

    start_time = time.time()

    # Call the dummy proofreading function
    result = dummy_proofread(input_path, edit_path, correction_path)

    wrapper = XmlPowerToolsEngine()
    output = wrapper.run_redline(
        author_tag="smart-docprocessor",
        original=input_path,
        modified=edit_path,
    )

    with open(tracked_correction_path, "wb") as f:
        f.write(output[0])

    elapsed_time = time.time() - start_time
    processing_time = f"{elapsed_time:.2f} seconds"

    return (
        f"<h3>Processing Completed in {processing_time}. Proceed to download your files.</h3>"
        f"<a href='/download/{filename}/edited'>Download Edited Document</a><br>"
        f"<a href='/download/{filename}/corrections'>Download Corrections Document</a>"
        f"<a href='/download/{filename}/tracked_corrections'>Download Tracked Corrections Document</a>"
    )


@app.route("/download/<filename>/<filetype>")
def download_file(filename, filetype):
    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"], filename.split(".")[0] + f"_{filetype}.docx"
    )
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(
        host=settings.flask.host,
        port=int(os.getenv("CDSW_READONLY_PORT", settings.flask.port)),
    )
