from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import openai
import uuid
from utils.ai_tutor import AITutor
from utils.whatsapp_notifier import WhatsAppNotifier
from utils.anti_cheat import AntiCheatMonitor
from models.database import db, Student, Admin, Course, Chapter, Exam, ExamResult, ChatSession, Notification

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///ai_school.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

# Initialize AI Tutor and WhatsApp Notifier
try:
    ai_tutor = AITutor(os.getenv('OPENAI_API_KEY'))
except Exception as e:
    print(f"Warning: AI Tutor initialization failed: {e}")
    ai_tutor = None

try:
    whatsapp_notifier = WhatsAppNotifier(
        os.getenv('TWILIO_ACCOUNT_SID'),
        os.getenv('TWILIO_AUTH_TOKEN'),
        os.getenv('TWILIO_WHATSAPP_FROM')
    )
except Exception as e:
    print(f"Warning: WhatsApp Notifier initialization failed: {e}")
    whatsapp_notifier = None

anti_cheat = AntiCheatMonitor()

@app.route('/')
def welcome():
    """Welcome page with 4 sections as requested"""
    return render_template('welcome.html')

@app.route('/student-login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        student = Student.query.filter_by(username=username).first()
        if student and check_password_hash(student.password_hash, password):
            session['user_id'] = student.id
            session['user_type'] = 'student'
            session['user_name'] = f"{student.first_name} {student.last_name}"
            return jsonify({'success': True, 'redirect': url_for('student_dashboard')})
        else:
            return jsonify({'success': False, 'message': 'نام کاربری یا رمز عبور اشتباه است'})
    
    return render_template('student_login.html')

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        admin = Admin.query.filter_by(username=username).first()
        if admin and check_password_hash(admin.password_hash, password):
            session['user_id'] = admin.id
            session['user_type'] = 'admin'
            session['user_name'] = admin.username
            return jsonify({'success': True, 'redirect': url_for('admin_dashboard')})
        else:
            return jsonify({'success': False, 'message': 'نام کاربری یا رمز عبور اشتباه است'})
    
    return render_template('admin_login.html')

@app.route('/student-dashboard')
def student_dashboard():
    if session.get('user_type') != 'student':
        return redirect(url_for('student_login'))
    
    student = Student.query.get(session['user_id'])
    courses = Course.query.filter_by(grade_level=student.grade_level).all()
    
    # Get student progress for each course
    course_progress = {}
    for course in courses:
        chapters = Chapter.query.filter_by(course_id=course.id).order_by(Chapter.order).all()
        unlocked_chapters = []
        
        for i, chapter in enumerate(chapters):
            if i == 0:  # First chapter is always unlocked
                unlocked_chapters.append(chapter.id)
            else:
                # Check if previous chapter exam was passed
                prev_chapter = chapters[i-1]
                exam_result = ExamResult.query.filter_by(
                    student_id=student.id,
                    chapter_id=prev_chapter.id
                ).order_by(ExamResult.created_at.desc()).first()
                
                if exam_result and exam_result.score >= 17:
                    unlocked_chapters.append(chapter.id)
        
        course_progress[course.id] = unlocked_chapters
    
    # Get recent notifications
    notifications = Notification.query.filter_by(student_id=student.id, read=False).order_by(Notification.created_at.desc()).limit(5).all()
    
    return render_template('student_dashboard.html', 
                         student=student, 
                         courses=courses, 
                         course_progress=course_progress,
                         notifications=notifications)

@app.route('/study/<int:course_id>/<int:chapter_id>')
def study_chapter(course_id, chapter_id):
    if session.get('user_type') != 'student':
        return redirect(url_for('student_login'))
    
    student = Student.query.get(session['user_id'])
    course = Course.query.get_or_404(course_id)
    chapter = Chapter.query.get_or_404(chapter_id)
    
    # Check if chapter is unlocked
    chapters = Chapter.query.filter_by(course_id=course_id).order_by(Chapter.order).all()
    chapter_index = next((i for i, c in enumerate(chapters) if c.id == chapter_id), None)
    
    if chapter_index > 0:
        prev_chapter = chapters[chapter_index - 1]
        exam_result = ExamResult.query.filter_by(
            student_id=student.id,
            chapter_id=prev_chapter.id
        ).order_by(ExamResult.created_at.desc()).first()
        
        if not exam_result or exam_result.score < 17:
            flash('باید ابتدا امتحان فصل قبلی را با نمره 17 یا بالاتر قبول شوید', 'warning')
            return redirect(url_for('student_dashboard'))
    
    # Get or create chat session for this chapter
    chat_session = ChatSession.query.filter_by(
        student_id=student.id,
        chapter_id=chapter_id
    ).first()
    
    if not chat_session:
        chat_session = ChatSession(
            student_id=student.id,
            chapter_id=chapter_id,
            messages=json.dumps([])
        )
        db.session.add(chat_session)
        db.session.commit()
    
    return render_template('study_chapter.html', 
                         student=student, 
                         course=course, 
                         chapter=chapter,
                         chat_session=chat_session)

@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    if session.get('user_type') != 'student':
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    message = data.get('message', '')
    chapter_id = data.get('chapter_id')
    
    if not chapter_id:
        return jsonify({'error': 'Chapter ID required'}), 400
    
    student = Student.query.get(session['user_id'])
    chapter = Chapter.query.get(chapter_id)
    
    # Get chat session
    chat_session = ChatSession.query.filter_by(
        student_id=student.id,
        chapter_id=chapter_id
    ).first()
    
    if not chat_session:
        return jsonify({'error': 'Chat session not found'}), 404
    
    # Check if student is requesting exam
    if 'امتحان' in message and ('بدم' in message or 'بگیرم' in message):
        return jsonify({
            'type': 'exam_request',
            'message': 'آیا آماده شرکت در امتحان این فصل هستید؟'
        })
    
    # Get AI response
    messages = json.loads(chat_session.messages)
    messages.append({'role': 'user', 'content': message, 'timestamp': datetime.utcnow().isoformat()})
    
    # Get AI response based on chapter content
    if ai_tutor:
        ai_response = ai_tutor.get_response(message, chapter.content, chapter.title)
    else:
        ai_response = f"متاسفانه سیستم هوش مصنوعی در حال حاضر در دسترس نیست. پیام شما: {message}"
    
    messages.append({'role': 'assistant', 'content': ai_response, 'timestamp': datetime.utcnow().isoformat()})
    
    # Update chat session
    chat_session.messages = json.dumps(messages)
    chat_session.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({
        'type': 'chat_response',
        'message': ai_response
    })

@app.route('/api/start-exam', methods=['POST'])
def start_exam():
    if session.get('user_type') != 'student':
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    chapter_id = data.get('chapter_id')
    
    student = Student.query.get(session['user_id'])
    chapter = Chapter.query.get(chapter_id)
    
    # Check exam attempts
    attempt_count = ExamResult.query.filter_by(
        student_id=student.id,
        chapter_id=chapter_id
    ).count()
    
    if attempt_count >= 3:
        last_result = ExamResult.query.filter_by(
            student_id=student.id,
            chapter_id=chapter_id
        ).order_by(ExamResult.created_at.desc()).first()
        
        if last_result.score < 17:
            return jsonify({
                'error': 'شما سه بار امتحان داده‌اید. برای امتحان مجدد باید با مدیریت تماس بگیرید'
            }), 400
    
    # Generate exam questions using AI
    if ai_tutor:
        questions = ai_tutor.generate_exam_questions(chapter.content, chapter.title)
    else:
        # Fallback questions for development
        questions = [
            {
                "id": 1,
                "type": "multiple_choice",
                "question": f"سوال نمونه درباره {chapter.title}",
                "options": ["گزینه الف", "گزینه ب", "گزینه ج", "گزینه د"],
                "correct_answer": 0,
                "points": 2
            },
            {
                "id": 2,
                "type": "essay",
                "question": f"سوال تشریحی درباره {chapter.title}",
                "points": 3,
                "sample_answer": "پاسخ نمونه"
            }
        ]
    
    # Create exam session
    exam_id = str(uuid.uuid4())
    session[f'exam_{exam_id}'] = {
        'chapter_id': chapter_id,
        'questions': questions,
        'start_time': datetime.utcnow().isoformat(),
        'duration': 90 * len(questions)  # 1.5 minutes per question
    }
    
    return jsonify({
        'exam_id': exam_id,
        'questions': questions,
        'duration': 90 * len(questions)
    })

@app.route('/api/submit-exam', methods=['POST'])
def submit_exam():
    if session.get('user_type') != 'student':
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    exam_id = data.get('exam_id')
    answers = data.get('answers')
    
    exam_data = session.get(f'exam_{exam_id}')
    if not exam_data:
        return jsonify({'error': 'Exam session not found'}), 404
    
    # Check if exam time has expired
    start_time = datetime.fromisoformat(exam_data['start_time'])
    if datetime.utcnow() > start_time + timedelta(seconds=exam_data['duration']):
        return jsonify({'error': 'زمان امتحان به پایان رسیده است'}), 400
    
    # Grade the exam
    questions = exam_data['questions']
    if ai_tutor:
        score = ai_tutor.grade_exam(questions, answers)
    else:
        # Fallback grading for development
        score = 15.0  # Default score for testing
    
    # Save exam result
    exam_result = ExamResult(
        student_id=session['user_id'],
        chapter_id=exam_data['chapter_id'],
        questions=json.dumps(questions),
        answers=json.dumps(answers),
        score=score,
        passed=score >= 17
    )
    db.session.add(exam_result)
    db.session.commit()
    
    # Send WhatsApp notification to parents
    student = Student.query.get(session['user_id'])
    chapter = Chapter.query.get(exam_data['chapter_id'])
    course = Course.query.get(chapter.course_id)
    
    if whatsapp_notifier:
        whatsapp_notifier.send_exam_result(
            student.parent_phone,
            student.first_name + ' ' + student.last_name,
            course.name,
            chapter.title,
            score
        )
    
    # Clean up exam session
    del session[f'exam_{exam_id}']
    
    return jsonify({
        'score': score,
        'passed': score >= 17,
        'message': 'تبریک! شما قبول شدید' if score >= 17 else 'متاسفانه نمره شما کافی نیست. نمره قبولی 17 است'
    })

@app.route('/admin-dashboard')
def admin_dashboard():
    if session.get('user_type') != 'admin':
        return redirect(url_for('admin_login'))
    
    # Get statistics
    total_students = Student.query.count()
    total_courses = Course.query.count()
    total_exams_today = ExamResult.query.filter(
        ExamResult.created_at >= datetime.utcnow().date()
    ).count()
    
    # Get recent activities
    recent_exams = db.session.query(ExamResult, Student, Chapter, Course).join(
        Student, ExamResult.student_id == Student.id
    ).join(
        Chapter, ExamResult.chapter_id == Chapter.id
    ).join(
        Course, Chapter.course_id == Course.id
    ).order_by(ExamResult.created_at.desc()).limit(10).all()
    
    return render_template('admin_dashboard.html',
                         total_students=total_students,
                         total_courses=total_courses,
                         total_exams_today=total_exams_today,
                         recent_exams=recent_exams)

@app.route('/admin/students')
def admin_students():
    if session.get('user_type') != 'admin':
        return redirect(url_for('admin_login'))
    
    students = Student.query.all()
    return render_template('admin_students.html', students=students)

@app.route('/admin/add-student', methods=['GET', 'POST'])
def add_student():
    if session.get('user_type') != 'admin':
        return redirect(url_for('admin_login'))
    
    if request.method == 'POST':
        data = request.get_json()
        
        # Check if username already exists
        if Student.query.filter_by(username=data['username']).first():
            return jsonify({'success': False, 'message': 'نام کاربری قبلاً استفاده شده است'})
        
        student = Student(
            username=data['username'],
            password_hash=generate_password_hash(data['password']),
            first_name=data['first_name'],
            last_name=data['last_name'],
            email=data['email'],
            phone=data['phone'],
            parent_phone=data['parent_phone'],
            grade_level=data['grade_level']
        )
        
        db.session.add(student)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'دانش‌آموز با موفقیت اضافه شد'})
    
    return render_template('add_student.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('welcome'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)