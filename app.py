from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory, send_file
import base64
import io
import os
from datetime import datetime
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from generatereport import generate_xray_report

from predit import Explanation

app = Flask(__name__, template_folder='htmls')

# Secret key for session management
app.secret_key = 'your_secret_key'

# Initialize the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chest.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

UPLOAD_FOLDER = 'public/dp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the Admin model
class Admin(db.Model):
    admin_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Define the Doctor model
class Doctor(db.Model):
    doctor_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    department = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(50), default='inactive')

# Create the tables and add initial data
def initialize_database():
    with app.app_context():
        db.create_all()

        # Check if the admin table is empty
        if not Admin.query.first():
            # Add initial admin record
            initial_admin = Admin(username='admin', password='admin')
            db.session.add(initial_admin)
            db.session.commit()

initialize_database()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_type = request.form.get('user_type')
        email = request.form.get('email')
        password = request.form.get('password')

        if user_type == 'admin':
            admin = Admin.query.filter_by(username=email).first()
            if admin and admin.password == password:  # Replace with hashed password check in production
                session['user_type'] = 'admin'
                session['user_id'] = admin.admin_id
                session['username'] = admin.username
                return redirect(url_for('admin_dashboard'))

        elif user_type == 'doctor':
            doctor = Doctor.query.filter_by(email=email).first()
            if doctor and check_password_hash(doctor.password, password):
                session['user_type'] = 'doctor'
                session['user_id'] = doctor.doctor_id
                session['email'] = doctor.email
                session['username'] = doctor.full_name
                return redirect(url_for('doctor_dashboard'))
            else:
                return "Invalid doctor credentials", 401

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        department = request.form.get('department')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        terms = request.form.get('terms')
        profile_picture = request.files.get('profile_picture')

        # Validation
        if not (full_name and email and department and password and confirm_password):
            return jsonify({"error": "All fields are required."}), 400

        if password != confirm_password:
            return jsonify({"error": "Passwords do not match."}), 400

        if not terms:
            return jsonify({"error": "You must agree to the terms and conditions."}), 400

        # Check if email already exists
        if Doctor.query.filter_by(email=email).first():
            return jsonify({"error": "Email already registered."}), 400

        # Create new doctor account (inactive until admin approval)
        hashed_password = generate_password_hash(password)
        new_doctor = Doctor(
            full_name=full_name,
            email=email,
            department=department,
            password=hashed_password,
            status='inactive'
        )
        
        try:
            db.session.add(new_doctor)
            db.session.commit()
            
            if profile_picture:
                filename = secure_filename(f"{new_doctor.doctor_id}.jpg")
                profile_picture.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return render_template('login.html', message="Registration successful! Please wait for admin approval.")
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": "An error occurred during registration."}), 500

    return render_template('login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get('user_type') == 'admin':
        return redirect(url_for('manage_doctors'))
    return redirect(url_for('login'))

@app.route('/doctor_dashboard')
def doctor_dashboard():
    if session.get('user_type') != 'doctor':
        return redirect(url_for('login'))
    return render_template('doctor.html', 
                         username=session.get('username'),
                         email=session.get('email'),
                         doctor_id=session.get('user_id'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def home():
    return "Image Classification and Explanation API"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'GET':
            # If accessed via GET, redirect to doctor dashboard
            return redirect(url_for('doctor_dashboard'))
            
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        image_file.seek(0)  # Reset file pointer
        returslt_image, labels = Explanation(image_file)
        imagebase64 = base64.b64encode(returslt_image.read()).decode('utf-8')
        returslt_image.close()
        
        # Sort labels to find the predicted class
        predicted_class = max(labels, key=labels.get)
        predicted_confidence = labels[predicted_class]
        
        # Create result data for template
        result_data = {
            'predicted_class': predicted_class,
            'confidence': predicted_confidence,
            'explanation': imagebase64,
            'labels': labels,
            'uid': f"#{str(session.get('user_id')).zfill(5)}-PX" if session.get('user_id') else "#00000-PX",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return render_template('result.html', result=result_data, doctor_id=session.get('user_id'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin/doctors', methods=['GET', 'POST'])
def manage_doctors():
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))

    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    if request.method == 'POST':
        # Add a new doctor
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        department = request.form.get('department')
        password = request.form.get('password')
        profile_picture = request.files.get('profile_picture')

        if not (full_name and email and department and password):
            return jsonify({"error": "All fields are required."}), 400

        # Check if email already exists
        if Doctor.query.filter_by(email=email).first():
            return jsonify({"error": "Doctor with this email already exists."}), 400

        hashed_password = generate_password_hash(password)
        new_doctor = Doctor(full_name=full_name, email=email, department=department, password=hashed_password)
        db.session.add(new_doctor)
        db.session.commit()

        # Save profile picture
        if profile_picture:
            filename = secure_filename(f"{new_doctor.doctor_id}.jpg")
            profile_picture.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return redirect(url_for('manage_doctors', page=1))

    # Fetch doctors with pagination
    pagination = Doctor.query.order_by(Doctor.doctor_id.desc()).paginate(page=page, per_page=per_page, error_out=False)
    doctors = pagination.items
    return render_template('admin.html', doctors=doctors, pagination=pagination, page=page, per_page=per_page, admin_username=session.get('username'))

@app.route('/public/dp/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/admin/doctors/edit/<int:doctor_id>', methods=['POST'])
def edit_doctor(doctor_id):
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))

    doctor = Doctor.query.get_or_404(doctor_id)
    doctor.full_name = request.form.get('full_name', doctor.full_name)
    doctor.email = request.form.get('email', doctor.email)
    doctor.department = request.form.get('department', doctor.department)
    doctor.status = request.form.get('status', doctor.status)
    if request.form.get('password'):
        doctor.password = generate_password_hash(request.form.get('password'))

    # Update profile picture
    profile_picture = request.files.get('profile_picture')
    if profile_picture:
        filename = secure_filename(f"{doctor.doctor_id}.jpg")
        profile_picture.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    db.session.commit()
    # preserve page if provided
    page = request.form.get('page', 1)
    try:
        page = int(page)
    except Exception:
        page = 1
    return redirect(url_for('manage_doctors', page=page))

@app.route('/admin/doctors/delete/<int:doctor_id>', methods=['POST'])
def delete_doctor(doctor_id):
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))

    doctor = Doctor.query.get_or_404(doctor_id)
    db.session.delete(doctor)
    db.session.commit()
    # preserve page if provided
    page = request.form.get('page', 1)
    try:
        page = int(page)
    except Exception:
        page = 1
    return redirect(url_for('manage_doctors', page=page))

@app.route('/generate_report_pdf/<int:doctor_id>', methods=['POST'])
def generate_report_pdf(doctor_id):
    if session.get('user_type') != 'doctor' or session.get('user_id') != doctor_id:
        return redirect(url_for('login'))

    try:
        # Get form data
        patient_name = request.form.get('patient_name')
        patient_age = request.form.get('patient_age')
        patient_gender = request.form.get('patient_gender')
        patient_id = request.form.get('patient_id')
        patient_phone = request.form.get('patient_phone')
        patient_email = request.form.get('patient_email')
        disease_name = request.form.get('disease_name')
        findings = request.form.get('findings')
        impression = request.form.get('impression')
        image_result_base64 = request.form.get('image_result_base64')  # Base64 explanation image from result page

        # Validate required fields
        if not (patient_name and patient_age and patient_gender and disease_name and findings and impression):
            return jsonify({"error": "All patient and report fields are required."}), 400

        # Generate PDF report with the explanation image
        report_path = generate_xray_report(
            hospital_name="University College Of Engineering",
            patient_info={
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "id": patient_id or f"#{str(doctor_id).zfill(5)}-PX",
                "phone": patient_phone,
                "email": patient_email,
                "findings": findings,
                "impression": impression
            },
            disease_name=disease_name,
            xray_image_base64=image_result_base64,  # Pass the explanation image as base64
            doctor_info={
                "name": session.get('username'),
                "email": session.get('email')
            }
        )

        return send_file(report_path, as_attachment=True, mimetype='application/pdf')
    
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return jsonify({"error": f"Error generating report: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)