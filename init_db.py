#!/usr/bin/env python3
"""
Database initialization script for AI School System
Creates tables and inserts sample data
"""

import os
import sys
from werkzeug.security import generate_password_hash
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
from models.database import db, Student, Admin, Course, Chapter, GradeLevel

def create_sample_data():
    """Create sample data for testing"""
    
    # Create grade levels
    grade_levels = [
        GradeLevel(name="نهم", description="پایه نهم دبیرستان"),
        GradeLevel(name="دهم", description="پایه دهم دبیرستان"),
        GradeLevel(name="یازدهم", description="پایه یازدهم دبیرستان"),
        GradeLevel(name="دوازدهم", description="پایه دوازدهم دبیرستان"),
    ]
    
    for grade in grade_levels:
        db.session.add(grade)
    
    # Create admin user
    admin = Admin(
        username="admin",
        password_hash=generate_password_hash("admin123"),
        full_name="مدیر سیستم",
        email="admin@ai-school.ir"
    )
    db.session.add(admin)
    
    # Create sample student
    student = Student(
        username="student1",
        password_hash=generate_password_hash("123456"),
        first_name="علی",
        last_name="احمدی",
        email="ali@example.com",
        phone="09123456789",
        parent_phone="09121234567",
        grade_level="نهم"
    )
    db.session.add(student)
    
    # Create sample courses for grade 9
    math_course = Course(
        name="ریاضی",
        description="دروس ریاضی پایه نهم شامل جبر، هندسه و آمار",
        grade_level="نهم"
    )
    
    science_course = Course(
        name="علوم تجربی",
        description="مفاهیم پایه فیزیک، شیمی و زیست‌شناسی",
        grade_level="نهم"
    )
    
    persian_course = Course(
        name="زبان فارسی",
        description="ادبیات، دستور زبان و املای فارسی",
        grade_level="نهم"
    )
    
    db.session.add_all([math_course, science_course, persian_course])
    db.session.commit()  # Commit to get course IDs
    
    # Create chapters for Math course
    math_chapters = [
        {
            "title": "اعداد طبیعی و مضربها",
            "content": """
            در این فصل با مفاهیم زیر آشنا خواهید شد:
            
            1. اعداد طبیعی: اعداد 1، 2، 3، 4، ... که برای شمارش استفاده می‌شوند
            2. مضرب عدد: اگر عدد a بر عدد b بخش‌پذیر باشد، گفته می‌شود a مضرب b است
            3. مضرب مشترک: عددی که مضرب چند عدد باشد
            4. کمترین مضرب مشترک (ک.م.م): کوچکترین عدد مثبتی که مضرب همه اعداد داده شده باشد
            
            مثال: ک.م.م اعداد 6 و 8 برابر 24 است.
            
            تمرین: ک.م.م اعداد زیر را پیدا کنید:
            - 4 و 6
            - 9 و 12
            - 15، 20 و 25
            """,
            "order": 1
        },
        {
            "title": "عددهای اول و تجزیه",
            "content": """
            در این فصل خواهید آموخت:
            
            1. عدد اول: عددی که فقط بر 1 و خودش بخش‌پذیر باشد
            2. عدد مرکب: عددی که علاوه بر 1 و خودش، مقسوم علیه‌های دیگری نیز داشته باشد
            3. تجزیه به عوامل اول: نوشتن هر عدد به صورت حاصل‌ضرب اعداد اول
            
            مثال: 60 = 2² × 3 × 5
            
            روش پیدا کردن اعداد اول تا 100:
            - غربال اراتوستن
            - تست تقسیم‌پذیری
            
            کاربردها:
            - محاسبه ب.م.م و ک.م.م
            - ساده کردن کسرها
            - حل مسائل ترکیبی
            """,
            "order": 2
        },
        {
            "title": "کسرها و اعمال روی آنها",
            "content": """
            مطالب این فصل:
            
            1. تعریف کسر: نسبت دو عدد صحیح که مخرج آن صفر نباشد
            2. انواع کسرها:
               - کسر حقیقی: صورت < مخرج
               - کسر غیرحقیقی: صورت ≥ مخرج
               - عدد مخلوط: شامل قسمت صحیح و کسری
            
            3. عملیات روی کسرها:
               - جمع و تفریق: مخرج مشترک
               - ضرب: ضرب صورتها در صورتها و مخرجها در مخرجها
               - تقسیم: ضرب در معکوس
            
            4. ساده کردن کسرها با استفاده از ب.م.م
            
            مثال: 3/4 + 2/6 = 9/12 + 4/12 = 13/12
            """,
            "order": 3
        }
    ]
    
    for i, chapter_data in enumerate(math_chapters):
        chapter = Chapter(
            course_id=math_course.id,
            title=chapter_data["title"],
            content=chapter_data["content"],
            order=chapter_data["order"]
        )
        db.session.add(chapter)
    
    # Create chapters for Science course
    science_chapters = [
        {
            "title": "آشنایی با ماده",
            "content": """
            در این فصل با مفاهیم اساسی ماده آشنا می‌شوید:
            
            1. تعریف ماده: هر چیزی که جرم داشته باشد و فضا را اشغال کند
            2. حالات ماده:
               - جامد: حجم و شکل ثابت
               - مایع: حجم ثابت، شکل متغیر
               - گاز: حجم و شکل متغیر
            
            3. خواص ماده:
               - خواص فیزیکی: رنگ، بو، طعم، چگالی
               - خواص شیمیایی: قابلیت سوختن، واکنش با اسید
            
            4. تغییرات ماده:
               - تغییرات فیزیکی: تغییر شکل یا حالت
               - تغییرات شیمیایی: تشکیل ماده جدید
            
            آزمایش: بررسی نقطه ذوب یخ و نقطه جوش آب
            """,
            "order": 1
        },
        {
            "title": "اتم و مولکول",
            "content": """
            ساختار بنیادی ماده:
            
            1. اتم: کوچکترین واحد ماده که خواص عنصر را دارد
               - هسته: شامل پروتون (+) و نوترون (0)
               - الکترون: ذرات منفی که دور هسته می‌چرخند
            
            2. عنصر: ماده‌ای که از یک نوع اتم تشکیل شده
               - 118 عنصر شناخته شده
               - نمادهای شیمیایی: H, O, C, N, ...
            
            3. مولکول: گروهی از اتم‌ها که با پیوند شیمیایی متصل شده‌اند
               - مولکول آب: H₂O
               - مولکول اکسیژن: O₂
            
            4. ترکیب: ماده‌ای که از دو یا چند عنصر تشکیل شده
            
            جدول تناوبی: ترتیب عناصر بر اساس عدد اتمی
            """,
            "order": 2
        }
    ]
    
    for chapter_data in science_chapters:
        chapter = Chapter(
            course_id=science_course.id,
            title=chapter_data["title"],
            content=chapter_data["content"],
            order=chapter_data["order"]
        )
        db.session.add(chapter)
    
    # Create chapters for Persian course
    persian_chapters = [
        {
            "title": "واژه و معنی",
            "content": """
            در این فصل با مباحث زیر آشنا می‌شوید:
            
            1. واژه: کوچکترین واحد معنادار زبان
            2. انواع واژه از نظر معنی:
               - واژه اصلی: معنی مستقل دارد
               - واژه کمکی: کمک به معنی‌سازی می‌کند
            
            3. روابط معنایی:
               - مترادف: واژگان هم‌معنی (بزرگ = کلان)
               - متضاد: واژگان مخالف (سیاه ≠ سفید)
               - هم‌آوا: تلفظ یکسان، معنی متفاوت (بار: محموله / بار: نوبت)
            
            4. ساخت واژه:
               - ریشه + پسوند = کتاب + خانه = کتابخانه
               - پیشوند + ریشه = نا + امید = ناامید
            
            تمرین: مترادف و متضاد کلمات زیر را بیابید:
            - شادی، غم، روشنی، تاریکی
            """,
            "order": 1
        }
    ]
    
    for chapter_data in persian_chapters:
        chapter = Chapter(
            course_id=persian_course.id,
            title=chapter_data["title"],
            content=chapter_data["content"],
            order=chapter_data["order"]
        )
        db.session.add(chapter)
    
    # Commit all changes
    db.session.commit()
    print("✅ Sample data created successfully!")

def init_database():
    """Initialize database with tables and sample data"""
    
    with app.app_context():
        print("🔄 Creating database tables...")
        
        # Drop all tables and recreate (for development)
        db.drop_all()
        db.create_all()
        print("✅ Database tables created successfully!")
        
        print("🔄 Creating sample data...")
        create_sample_data()
        
        print("🎉 Database initialization completed!")
        print("\n📝 Sample login credentials:")
        print("Admin - Username: admin, Password: admin123")
        print("Student - Username: student1, Password: 123456")
        print("\n🌐 Run 'python app.py' to start the server")

if __name__ == "__main__":
    init_database()