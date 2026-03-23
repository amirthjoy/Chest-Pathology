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
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit for large medical images
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


class ReportLog(db.Model):
    log_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.doctor_id'), nullable=False)
    doctor_name = db.Column(db.String(100), nullable=False)
    doctor_email = db.Column(db.String(100), nullable=False)
    patient_name = db.Column(db.String(150), nullable=False)
    patient_age = db.Column(db.String(20), nullable=False)
    patient_gender = db.Column(db.String(20), nullable=False)
    patient_id = db.Column(db.String(100))
    patient_phone = db.Column(db.String(50))
    patient_email = db.Column(db.String(100))
    disease_name = db.Column(db.String(150), nullable=False)
    findings = db.Column(db.Text, nullable=False)
    impression = db.Column(db.Text, nullable=False)
    explanation_filename = db.Column(db.String(255))
    report_path = db.Column(db.String(255))
    ip_address = db.Column(db.String(64))
    device_info = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class SecurityLog(db.Model):
    security_log_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.doctor_id'))
    doctor_name = db.Column(db.String(100))
    doctor_email = db.Column(db.String(100))
    event_type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    ip_address = db.Column(db.String(64))
    device_info = db.Column(db.String(255))
    user_agent = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    notes = db.Column(db.Text)


def get_client_ip():
    forwarded_for = request.headers.get('X-Forwarded-For', '')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    return request.remote_addr or 'Unknown'


def get_device_info():
    user_agent = request.user_agent
    parts = [user_agent.platform, user_agent.browser]
    details = " / ".join(part for part in parts if part)
    return details or (request.headers.get('User-Agent', 'Unknown device')[:255])


def create_security_log(event_type, status, doctor=None, email=None, notes=None):
    security_log = SecurityLog(
        doctor_id=doctor.doctor_id if doctor else None,
        doctor_name=doctor.full_name if doctor else None,
        doctor_email=(doctor.email if doctor else email),
        event_type=event_type,
        status=status,
        ip_address=get_client_ip(),
        device_info=get_device_info(),
        user_agent=request.headers.get('User-Agent', '')[:500],
        notes=notes
    )
    db.session.add(security_log)
    db.session.commit()

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
                create_security_log('login', 'success', doctor=doctor)
                return redirect(url_for('doctor_dashboard'))
            else:
                create_security_log('login', 'failed', doctor=doctor, email=email, notes='Invalid doctor credentials')
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

@app.route('/')
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
    if session.get('user_type') == 'doctor':
        doctor = Doctor.query.get(session.get('user_id'))
        create_security_log('logout', 'success', doctor=doctor, email=session.get('email'))
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def home():
    return "Image Classification and Explanation API"

def compress_image(input_path, quality=70, max_width=None, max_height=None):
    image = Image.open(input_path)
    print("before compression:", image.size)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    if max_width or max_height:
        image.thumbnail((max_width or image.width, max_height or image.height))
    
    # Save to BytesIO object
    output = io.BytesIO()
    image.save(output, format='JPEG', optimize=True, quality=quality)
    output.seek(0)
    print("after compression:", image.size)
    return output

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'GET':
            # If accessed via GET, redirect to doctor dashboard
            return redirect(url_for('doctor_dashboard'))
            
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files['image']
        image_file = compress_image(image_file,max_height=512,max_width=512)  # Compress the image before processing
        image_file.seek(0)  # Reset file pointer
        returslt_image, labels = Explanation(image_file)
        
        # Save explanation image to temporary folder instead of encoding as base64
        temp_folder = 'public/explanations'
        os.makedirs(temp_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_id = session.get('user_id') or 0
        explanation_filename = f"explanation_{user_id}_{timestamp}.png"
        explanation_filepath = os.path.join(temp_folder, explanation_filename)
        
        # Write the image to file
        with open(explanation_filepath, 'wb') as f:
            f.write(returslt_image.read())
        returslt_image.close()
        
        # Sort labels to find the predicted class
        predicted_class = max(labels, key=labels.get)
        predicted_confidence = labels[predicted_class]
        
        # Create result data for template - pass filename instead of base64
        result_data = {
            'predicted_class': predicted_class,
            'confidence': predicted_confidence,
            'explanation': explanation_filename,  # Pass filename instead of base64
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
    return render_template(
        'admin.html',
        doctors=doctors,
        pagination=pagination,
        page=page,
        per_page=per_page,
        admin_username=session.get('username'),
        current_page='doctors'
    )


@app.route('/admin/logs')
def admin_logs():
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    pagination = ReportLog.query.order_by(ReportLog.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template(
        'admin_logs.html',
        logs=pagination.items,
        pagination=pagination,
        per_page=per_page,
        admin_username=session.get('username'),
        current_page='logs'
    )


@app.route('/admin/security-logs')
@app.route('/admin/secuity-logs')
def admin_security_logs():
    if session.get('user_type') != 'admin':
        return redirect(url_for('login'))

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    pagination = SecurityLog.query.order_by(SecurityLog.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template(
        'admin_security_logs.html',
        logs=pagination.items,
        pagination=pagination,
        per_page=per_page,
        admin_username=session.get('username'),
        current_page='security'
    )

@app.route('/public/dp/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/public/explanations/<filename>')
def explanation_file(filename):
    return send_from_directory('public/explanations', filename)

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
        explanation_filename = request.form.get('explanation_filename')  # Explanation image filename from result page

        # Validate required fields
        if not (patient_name and patient_age and patient_gender and disease_name and findings and impression):
            return jsonify({"error": "All patient and report fields are required."}), 400

        # Build path to explanation image
        explanation_image_path = None
        if explanation_filename:
            explanation_image_path = os.path.join('public/explanations', explanation_filename)
            if not os.path.exists(explanation_image_path):
                explanation_image_path = None

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
            xray_image_path=explanation_image_path,  # Pass file path instead of base64
            doctor_info={
                "name": session.get('username'),
                "email": session.get('email')
            }
        )

        report_log = ReportLog(
            doctor_id=doctor_id,
            doctor_name=session.get('username', ''),
            doctor_email=session.get('email', ''),
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            patient_id=patient_id or f"#{str(doctor_id).zfill(5)}-PX",
            patient_phone=patient_phone,
            patient_email=patient_email,
            disease_name=disease_name,
            findings=findings,
            impression=impression,
            explanation_filename=explanation_filename,
            report_path=report_path,
            ip_address=get_client_ip(),
            device_info=get_device_info()
        )
        db.session.add(report_log)
        db.session.commit()

        return send_file(report_path, as_attachment=True, mimetype='application/pdf')
    
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return jsonify({"error": f"Error generating report: {str(e)}"}), 500

if __name__ == '__main__':
    from waitress import serve
    print("Starting server on http://localhost:8000")
    serve(
        app,
        port=8000,
        max_request_body_size=100 * 1024 * 1024,  # 100 MB limit for large medical images
        inbuf_overflow=100 * 1024 * 1024,
        outbuf_overflow=100 * 1024 * 1024
    )
    # app.run(debug=True, port=8000)
