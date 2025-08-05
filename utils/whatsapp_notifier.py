from twilio.rest import Client
from twilio.base.exceptions import TwilioException
import logging

class WhatsAppNotifier:
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.logger = logging.getLogger(__name__)
    
    def send_exam_result(self, parent_phone: str, student_name: str, course_name: str, chapter_title: str, score: float) -> bool:
        """Send exam result notification to parent via WhatsApp"""
        try:
            # Format phone number for WhatsApp
            if not parent_phone.startswith('whatsapp:'):
                if parent_phone.startswith('+98'):
                    formatted_phone = f"whatsapp:{parent_phone}"
                elif parent_phone.startswith('09'):
                    formatted_phone = f"whatsapp:+98{parent_phone[1:]}"
                else:
                    formatted_phone = f"whatsapp:+98{parent_phone}"
            else:
                formatted_phone = parent_phone
            
            # Create message
            status_emoji = "✅" if score >= 17 else "❌"
            status_text = "قبول" if score >= 17 else "مردود"
            
            message = f"""
{status_emoji} نتیجه امتحان دانش‌آموز

👤 نام دانش‌آموز: {student_name}
📚 درس: {course_name}
📖 فصل: {chapter_title}
📊 نمره: {score}/20
📈 وضعیت: {status_text}

⏰ تاریخ: {self._get_persian_date()}

🏫 سیستم مدرسه آنلاین هوش مصنوعی
            """.strip()
            
            # Send message
            message_obj = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=formatted_phone
            )
            
            self.logger.info(f"WhatsApp message sent successfully to {formatted_phone}. SID: {message_obj.sid}")
            return True
            
        except TwilioException as e:
            self.logger.error(f"Twilio error sending WhatsApp message: {e}")
            return False
        except Exception as e:
            self.logger.error(f"General error sending WhatsApp message: {e}")
            return False
    
    def send_notification(self, parent_phone: str, student_name: str, title: str, message: str) -> bool:
        """Send general notification to parent via WhatsApp"""
        try:
            # Format phone number for WhatsApp
            if not parent_phone.startswith('whatsapp:'):
                if parent_phone.startswith('+98'):
                    formatted_phone = f"whatsapp:{parent_phone}"
                elif parent_phone.startswith('09'):
                    formatted_phone = f"whatsapp:+98{parent_phone[1:]}"
                else:
                    formatted_phone = f"whatsapp:+98{parent_phone}"
            else:
                formatted_phone = parent_phone
            
            # Create message
            full_message = f"""
📢 {title}

👤 دانش‌آموز: {student_name}
📝 پیام: {message}

⏰ تاریخ: {self._get_persian_date()}

🏫 سیستم مدرسه آنلاین هوش مصنوعی
            """.strip()
            
            # Send message
            message_obj = self.client.messages.create(
                body=full_message,
                from_=self.from_number,
                to=formatted_phone
            )
            
            self.logger.info(f"WhatsApp notification sent successfully to {formatted_phone}. SID: {message_obj.sid}")
            return True
            
        except TwilioException as e:
            self.logger.error(f"Twilio error sending WhatsApp notification: {e}")
            return False
        except Exception as e:
            self.logger.error(f"General error sending WhatsApp notification: {e}")
            return False
    
    def send_welcome_message(self, parent_phone: str, student_name: str, username: str, password: str) -> bool:
        """Send welcome message when student account is created"""
        try:
            # Format phone number for WhatsApp
            if not parent_phone.startswith('whatsapp:'):
                if parent_phone.startswith('+98'):
                    formatted_phone = f"whatsapp:{parent_phone}"
                elif parent_phone.startswith('09'):
                    formatted_phone = f"whatsapp:+98{parent_phone[1:]}"
                else:
                    formatted_phone = f"whatsapp:+98{parent_phone}"
            else:
                formatted_phone = parent_phone
            
            # Create message
            message = f"""
🎉 خوش آمدید به مدرسه آنلاین هوش مصنوعی

👤 نام دانش‌آموز: {student_name}
🔐 اطلاعات ورود:
   نام کاربری: {username}
   رمز عبور: {password}

🌐 آدرس سایت: [آدرس سایت شما]

⚠️ لطفاً اطلاعات ورود را محفوظ نگه دارید.

⏰ تاریخ ثبت‌نام: {self._get_persian_date()}

🏫 سیستم مدرسه آنلاین هوش مصنوعی
            """.strip()
            
            # Send message
            message_obj = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=formatted_phone
            )
            
            self.logger.info(f"Welcome WhatsApp message sent successfully to {formatted_phone}. SID: {message_obj.sid}")
            return True
            
        except TwilioException as e:
            self.logger.error(f"Twilio error sending welcome WhatsApp message: {e}")
            return False
        except Exception as e:
            self.logger.error(f"General error sending welcome WhatsApp message: {e}")
            return False
    
    def _get_persian_date(self) -> str:
        """Get current date in Persian format"""
        from datetime import datetime
        import locale
        
        try:
            # Try to use Persian locale if available
            locale.setlocale(locale.LC_TIME, 'fa_IR.UTF-8')
        except:
            pass
        
        now = datetime.now()
        return now.strftime("%Y/%m/%d %H:%M")
    
    def test_connection(self) -> bool:
        """Test WhatsApp connection"""
        try:
            # Get account info to test connection
            account = self.client.api.accounts(self.client.account_sid).fetch()
            self.logger.info(f"WhatsApp connection test successful. Account: {account.friendly_name}")
            return True
        except Exception as e:
            self.logger.error(f"WhatsApp connection test failed: {e}")
            return False