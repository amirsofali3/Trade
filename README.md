# AI Online School System

This is a comprehensive AI-powered online school system with advanced features for student learning, exam management, and administrative control.

## Features

- **Student Portal**: AI-powered learning with course progression
- **Admin Panel**: Complete student and course management
- **AI Tutoring**: Intelligent chatbot for personalized learning
- **Exam System**: Anti-cheating mechanisms with timed assessments
- **Parent Notifications**: WhatsApp integration for grade updates
- **Progress Tracking**: Detailed analytics and PDF reports

## System Requirements

- Python 3.8+
- MySQL 8.0+
- OpenAI API Key
- Twilio Account (for WhatsApp)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file:
```
DATABASE_URL=mysql+pymysql://user:password@localhost/ai_school
OPENAI_API_KEY=your_openai_api_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
SECRET_KEY=your_secret_key
```

3. Initialize database:
```bash
python init_db.py
```

4. Run the application:
```bash
python app.py
```

## Usage

- Access the welcome page at `http://localhost:5000`
- Students can login and access AI-powered learning modules
- Admins can manage students, courses, and view analytics
- All data is stored securely in MySQL database

## Architecture

- **Frontend**: HTML/CSS/JavaScript with responsive design
- **Backend**: Flask with SQLAlchemy ORM
- **Database**: MySQL with comprehensive schema
- **AI Integration**: OpenAI GPT for intelligent tutoring
- **Notifications**: Twilio WhatsApp API