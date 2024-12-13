import os
import jwt as pyjwt
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from google.cloud import firestore
from Object_Detection.utils.object_localization import ocr_receipt
from recommender.full_deployment import full_deployment
import re
from functools import wraps
import dotenv

dotenv.load_dotenv()

app = Flask(__name__)

MODEL_PATH = "Object_Detection/Saved_Models/model.keras"
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

db = firestore.Client()
COLLECTION_NAME = "ocr_receipts"
JWT_SECRET = os.getenv("JWT_SECRET", "capstone_bangkit")
JWT_ALGORITHM = "HS256"

# Load TensorFlow model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Utility functions
def allowed_file(filename):
    """
    Validate if the uploaded file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def authenticate_request(func):
    """
    Middleware to authenticate requests using JWT.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "No token provided"}), 403
        try:
            # Decode JWT token
            decoded = pyjwt.decode(token.split(" ")[1], JWT_SECRET, algorithms=[JWT_ALGORITHM])
            request.uid = decoded.get("userId")  # Extract UID from token
            return func(*args, **kwargs)
        except pyjwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired. Please log in again."}), 401
        except pyjwt.InvalidTokenError:
            return jsonify({"error": "Invalid token."}), 401

    return wrapper


def extract_total_amount(extracted_text):
    """
    Extracts the total amount from the extracted text.

    Prioritize 'Total' over 'Subtotal' if both are present.

    Args:
        extracted_text (str): The OCR extracted text.

    Returns:
        str: The total amount or a message if not found.
    """
    normalized_text = extracted_text.replace(",", ".").lower()

    total_pattern = r"total[:\s]*\$?(\d+\.\d{2})"
    subtotal_pattern = r"subtotal[:\s]*\$?(\d+\.\d{2})"

    total_match = re.search(total_pattern, normalized_text)
    if total_match:
        return total_match.group(1)

    subtotal_match = re.search(subtotal_pattern, normalized_text)
    if subtotal_match:
        return subtotal_match.group(1)

    return "Total amount not found"

# API Endpoints
@app.route('/ocr', methods=['POST'])
@authenticate_request
def ocr_receipt_api():
    """
    API endpoint to perform OCR on uploaded receipts and store results in Firestore.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"File uploaded: {file_path}")

        try:
            extracted_text_raw = ocr_receipt(file_path, model)
            extracted_text_clean = extracted_text_raw.strip()

            lines = [line.strip() for line in extracted_text_clean.split("\n") if line.strip()]
            labeled_lines = {f"line_{idx + 1}": line for idx, line in enumerate(lines)}

            total_amount = extract_total_amount("\n".join(lines))

            record = {
                "filename": filename,
                "extracted_text": labeled_lines,
                "total_amount": total_amount,
                "uid": request.uid,  # Store user UID
            }

            db.collection(COLLECTION_NAME).add(record)

            return jsonify(record), 200
        except Exception as e:
            print(f"Error during OCR: {e}")
            return jsonify({"error": f"Error during OCR: {str(e)}"}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return jsonify({"error": "Invalid file type"}), 400


@app.route('/full-deployment', methods=['POST'])
@authenticate_request
def full_deployment_api():
    """
    API endpoint where the user uploads a photo and optionally provides longitude and latitude.
    """
    if not request.uid:
        return jsonify({"error": "UID is missing"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Extract file
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"File uploaded: {file_path}")

        try:
            # Fetch email from Firestore using UID
            user_ref = db.collection("users").document(request.uid)
            user_doc = user_ref.get()

            if not user_doc.exists:
                return jsonify({"error": "User not found"}), 404

            user_data = user_doc.to_dict()
            email = user_data.get("email")

            # Extract lon and lat from form data
            lon = request.form.get("lon", 106.8272)  # Default to Jakarta's longitude
            lat = request.form.get("lat", -6.1751)   # Default to Jakarta's latitude

            # Validate lon and lat
            try:
                lon = float(lon)
                lat = float(lat)
                if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                    return jsonify({"error": "Invalid longitude or latitude values."}), 400
            except ValueError:
                return jsonify({"error": "Longitude and latitude must be numeric."}), 400

            # Define other parameters
            key_path = os.getenv("GOOGLE_KEY_PATH", "key.json")
            dataset_path = os.getenv("DATASET_PATH", "dataset.csv")

            # Perform full deployment
            recommendations = full_deployment(
                key_path=key_path,
                test_path=file_path,
                dataset_path=dataset_path,
                uid=request.uid,
                email=email,
                model=model,
                lon=lon,
                lat=lat,
            )

            result = recommendations.to_dict(orient="records")
            return jsonify({"status": "success", "recommendations": result}), 200
        except Exception as e:
            print(f"Error during processing: {e}")
            return jsonify({"error": f"Error during processing: {str(e)}"}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return jsonify({"error": "Invalid file type"}), 400


@app.route('/records', methods=['GET'])
@authenticate_request
def get_records():
    """
    API endpoint to fetch all stored OCR records from Firestore.
    """
    try:
        records = db.collection(COLLECTION_NAME).where("uid", "==", request.uid).stream()
        result = [{"id": record.id, **record.to_dict()} for record in records]
        return jsonify(result), 200
    except Exception as e:
        print(f"Error fetching records: {e}")
        return jsonify({"error": f"Error fetching records: {str(e)}"}), 500


@app.route('/')
def index():
    """
    Health check endpoint.
    """
    return "OCR Receipt API is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
