import openai
import json
import random
from typing import List, Dict, Any

class AITutor:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
    def get_response(self, user_message: str, chapter_content: str, chapter_title: str) -> str:
        """Get AI response for student question based on chapter content"""
        try:
            system_prompt = f"""
            شما یک معلم هوش مصنوعی هستید که دانش‌آموزان ایرانی را در درس "{chapter_title}" راهنمایی می‌کنید.
            
            محتوای فصل:
            {chapter_content}
            
            وظایف شما:
            1. به سوالات دانش‌آموزان بر اساس محتوای فصل پاسخ دهید
            2. توضیحات ساده و قابل فهم ارائه دهید
            3. در صورت نیاز مثال‌های عملی بزنید
            4. اگر سوال خارج از محدوده فصل است، دانش‌آموز را به محتوای مرتبط هدایت کنید
            5. با احترام و صبر پاسخ دهید
            
            از زبان فارسی استفاده کنید و پاسخ‌هایتان مناسب برای سطح دانش‌آموزان باشد.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"متاسفانه خطایی رخ داده است. لطفاً دوباره تلاش کنید. خطا: {str(e)}"
    
    def generate_exam_questions(self, chapter_content: str, chapter_title: str, num_questions: int = 10) -> List[Dict[str, Any]]:
        """Generate exam questions based on chapter content"""
        try:
            system_prompt = f"""
            شما باید برای فصل "{chapter_title}" آزمون تولید کنید.
            
            محتوای فصل:
            {chapter_content}
            
            دستورالعمل:
            1. {num_questions} سوال تولید کنید
            2. 70% سوالات چهارگزینه‌ای و 30% تشریحی باشد
            3. سوالات باید از محتوای فصل باشد
            4. سطح سوالات متناسب با مقطع تحصیلی باشد
            5. هر سوال چهارگزینه‌ای باید دقیقاً یک گزینه صحیح داشته باشد
            
            فرمت خروجی JSON:
            {{
                "questions": [
                    {{
                        "id": 1,
                        "type": "multiple_choice",
                        "question": "متن سوال",
                        "options": ["گزینه الف", "گزینه ب", "گزینه ج", "گزینه د"],
                        "correct_answer": 0,
                        "points": 2
                    }},
                    {{
                        "id": 2,
                        "type": "essay",
                        "question": "متن سوال تشریحی",
                        "points": 3,
                        "sample_answer": "نمونه پاسخ برای تصحیح"
                    }}
                ]
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                max_tokens=2000,
                temperature=0.8
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return result["questions"]
            
        except Exception as e:
            # Fallback: Generate basic questions
            return self._generate_fallback_questions(chapter_title, num_questions)
    
    def _generate_fallback_questions(self, chapter_title: str, num_questions: int) -> List[Dict[str, Any]]:
        """Generate basic fallback questions if AI fails"""
        questions = []
        
        for i in range(num_questions):
            if i < num_questions * 0.7:  # Multiple choice
                questions.append({
                    "id": i + 1,
                    "type": "multiple_choice",
                    "question": f"سوال {i + 1} در مورد {chapter_title}",
                    "options": ["گزینه الف", "گزینه ب", "گزینه ج", "گزینه د"],
                    "correct_answer": 0,
                    "points": 2
                })
            else:  # Essay
                questions.append({
                    "id": i + 1,
                    "type": "essay",
                    "question": f"سوال تشریحی {i + 1} در مورد {chapter_title}",
                    "points": 3,
                    "sample_answer": "پاسخ نمونه"
                })
        
        return questions
    
    def grade_exam(self, questions: List[Dict[str, Any]], answers: Dict[str, Any]) -> float:
        """Grade exam and return score out of 20"""
        try:
            total_points = sum(q["points"] for q in questions)
            earned_points = 0
            
            for question in questions:
                q_id = str(question["id"])
                
                if question["type"] == "multiple_choice":
                    if q_id in answers and answers[q_id] == question["correct_answer"]:
                        earned_points += question["points"]
                        
                elif question["type"] == "essay":
                    if q_id in answers and answers[q_id].strip():
                        # Use AI to grade essay questions
                        essay_score = self._grade_essay_question(
                            question["question"],
                            question["sample_answer"],
                            answers[q_id],
                            question["points"]
                        )
                        earned_points += essay_score
            
            # Convert to 20-point scale
            score = (earned_points / total_points) * 20
            return min(20, max(0, round(score, 2)))
            
        except Exception as e:
            # Fallback scoring
            return 10.0  # Give average score if grading fails
    
    def _grade_essay_question(self, question: str, sample_answer: str, student_answer: str, max_points: int) -> float:
        """Grade essay question using AI"""
        try:
            system_prompt = f"""
            شما یک معلم هستید که باید سوال تشریحی را نمره‌دهی کنید.
            
            سوال: {question}
            پاسخ نمونه: {sample_answer}
            پاسخ دانش‌آموز: {student_answer}
            حداکثر امتیاز: {max_points}
            
            لطفاً پاسخ دانش‌آموز را بر اساس پاسخ نمونه نمره‌دهی کنید.
            فقط عدد امتیاز (بین 0 تا {max_points}) را برگردانید.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                max_tokens=10,
                temperature=0.3
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return min(max_points, max(0, score))
            
        except Exception as e:
            # Fallback: Give partial credit if answer exists
            return max_points * 0.5 if student_answer.strip() else 0
    
    def get_chapter_summary(self, chapter_content: str, chapter_title: str) -> str:
        """Generate chapter summary for student"""
        try:
            system_prompt = f"""
            لطفاً خلاصه‌ای از فصل "{chapter_title}" تهیه کنید.
            
            محتوای فصل:
            {chapter_content}
            
            خلاصه باید:
            1. نکات کلیدی را شامل شود
            2. ساده و قابل فهم باشد
            3. در حدود 200-300 کلمه باشد
            4. به زبان فارسی باشد
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                max_tokens=400,
                temperature=0.5
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"خلاصه فصل {chapter_title} در دسترس نیست."