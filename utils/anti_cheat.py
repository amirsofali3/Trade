from datetime import datetime
import logging
from typing import Dict, List

class AntiCheatMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_exams: Dict[str, Dict] = {}
        self.cheat_attempts: Dict[str, List] = {}
    
    def start_exam_monitoring(self, exam_id: str, student_id: int, duration_seconds: int) -> None:
        """Start monitoring an exam session for cheating attempts"""
        self.active_exams[exam_id] = {
            'student_id': student_id,
            'start_time': datetime.utcnow(),
            'duration_seconds': duration_seconds,
            'warnings': 0,
            'terminated': False
        }
        
        self.cheat_attempts[exam_id] = []
        self.logger.info(f"Started anti-cheat monitoring for exam {exam_id}, student {student_id}")
    
    def record_cheat_attempt(self, exam_id: str, attempt_type: str, details: str = "") -> Dict[str, any]:
        """Record a cheating attempt and determine action"""
        if exam_id not in self.active_exams:
            return {'action': 'ignore', 'message': 'Exam not found'}
        
        exam_info = self.active_exams[exam_id]
        
        if exam_info['terminated']:
            return {'action': 'ignore', 'message': 'Exam already terminated'}
        
        # Record the attempt
        attempt = {
            'type': attempt_type,
            'timestamp': datetime.utcnow(),
            'details': details
        }
        
        self.cheat_attempts[exam_id].append(attempt)
        exam_info['warnings'] += 1
        
        self.logger.warning(f"Cheat attempt detected - Exam: {exam_id}, Type: {attempt_type}, Details: {details}")
        
        # Determine action based on attempt type and count
        if attempt_type in ['window_blur', 'tab_switch', 'visibility_change']:
            if exam_info['warnings'] == 1:
                return {
                    'action': 'warning',
                    'message': 'هشدار: تغییر پنجره یا تب در حین امتحان مجاز نیست. در صورت تکرار، امتحان لغو خواهد شد.'
                }
            elif exam_info['warnings'] >= 2:
                exam_info['terminated'] = True
                return {
                    'action': 'terminate',
                    'message': 'امتحان به دلیل تقلب لغو شد. نمره شما صفر ثبت خواهد شد.',
                    'score': 0
                }
        
        elif attempt_type in ['copy_paste', 'right_click', 'developer_tools']:
            # Immediate termination for serious violations
            exam_info['terminated'] = True
            return {
                'action': 'terminate',
                'message': 'امتحان به دلیل استفاده از ابزارهای غیرمجاز لغو شد. نمره شما صفر ثبت خواهد شد.',
                'score': 0
            }
        
        elif attempt_type == 'time_exceeded':
            exam_info['terminated'] = True
            return {
                'action': 'time_up',
                'message': 'زمان امتحان به پایان رسیده است. پاسخ‌های داده شده محاسبه خواهد شد.'
            }
        
        return {'action': 'continue', 'message': 'ادامه امتحان'}
    
    def is_exam_terminated(self, exam_id: str) -> bool:
        """Check if exam has been terminated due to cheating"""
        if exam_id not in self.active_exams:
            return False
        return self.active_exams[exam_id]['terminated']
    
    def get_exam_warnings(self, exam_id: str) -> int:
        """Get number of warnings for an exam"""
        if exam_id not in self.active_exams:
            return 0
        return self.active_exams[exam_id]['warnings']
    
    def get_cheat_attempts(self, exam_id: str) -> List[Dict]:
        """Get all cheat attempts for an exam"""
        return self.cheat_attempts.get(exam_id, [])
    
    def end_exam_monitoring(self, exam_id: str) -> Dict[str, any]:
        """End monitoring and return summary"""
        if exam_id not in self.active_exams:
            return {'found': False}
        
        exam_info = self.active_exams[exam_id]
        attempts = self.cheat_attempts.get(exam_id, [])
        
        summary = {
            'found': True,
            'student_id': exam_info['student_id'],
            'start_time': exam_info['start_time'],
            'warnings': exam_info['warnings'],
            'terminated': exam_info['terminated'],
            'total_attempts': len(attempts),
            'attempt_types': list(set(attempt['type'] for attempt in attempts))
        }
        
        # Clean up
        del self.active_exams[exam_id]
        del self.cheat_attempts[exam_id]
        
        self.logger.info(f"Ended anti-cheat monitoring for exam {exam_id}. Summary: {summary}")
        return summary
    
    def check_time_limit(self, exam_id: str) -> Dict[str, any]:
        """Check if exam time limit has been exceeded"""
        if exam_id not in self.active_exams:
            return {'action': 'ignore', 'message': 'Exam not found'}
        
        exam_info = self.active_exams[exam_id]
        elapsed_time = (datetime.utcnow() - exam_info['start_time']).total_seconds()
        
        if elapsed_time > exam_info['duration_seconds']:
            return self.record_cheat_attempt(exam_id, 'time_exceeded', f'Elapsed: {elapsed_time}s')
        
        return {'action': 'continue', 'remaining_time': exam_info['duration_seconds'] - elapsed_time}
    
    def get_javascript_monitoring_code(self) -> str:
        """Return JavaScript code for client-side monitoring"""
        return """
        // Anti-cheat monitoring system
        (function() {
            let warnings = 0;
            let examTerminated = false;
            
            // Track window blur/focus
            window.addEventListener('blur', function() {
                if (!examTerminated) {
                    reportCheatAttempt('window_blur', 'Window lost focus');
                }
            });
            
            // Track visibility change
            document.addEventListener('visibilitychange', function() {
                if (document.hidden && !examTerminated) {
                    reportCheatAttempt('visibility_change', 'Page became hidden');
                }
            });
            
            // Disable right-click
            document.addEventListener('contextmenu', function(e) {
                e.preventDefault();
                if (!examTerminated) {
                    reportCheatAttempt('right_click', 'Right-click attempted');
                }
            });
            
            // Disable common keyboard shortcuts
            document.addEventListener('keydown', function(e) {
                // Disable F12, Ctrl+Shift+I, Ctrl+U, etc.
                if (e.key === 'F12' || 
                    (e.ctrlKey && e.shiftKey && e.key === 'I') ||
                    (e.ctrlKey && e.key === 'u') ||
                    (e.ctrlKey && e.shiftKey && e.key === 'C')) {
                    e.preventDefault();
                    if (!examTerminated) {
                        reportCheatAttempt('developer_tools', 'Developer tools access attempted');
                    }
                }
                
                // Disable copy/paste
                if (e.ctrlKey && (e.key === 'c' || e.key === 'v' || e.key === 'x')) {
                    e.preventDefault();
                    if (!examTerminated) {
                        reportCheatAttempt('copy_paste', 'Copy/paste attempted');
                    }
                }
            });
            
            function reportCheatAttempt(type, details) {
                fetch('/api/cheat-attempt', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        exam_id: window.currentExamId,
                        type: type,
                        details: details
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.action === 'warning') {
                        warnings++;
                        alert(data.message);
                    } else if (data.action === 'terminate') {
                        examTerminated = true;
                        alert(data.message);
                        window.location.href = '/student-dashboard';
                    }
                })
                .catch(error => console.error('Error reporting cheat attempt:', error));
            }
            
            // Make reportCheatAttempt available globally for manual calls
            window.reportCheatAttempt = reportCheatAttempt;
        })();
        """
    
    def get_exam_timer_code(self, duration_seconds: int) -> str:
        """Return JavaScript code for exam timer"""
        return f"""
        // Exam timer
        (function() {{
            let timeRemaining = {duration_seconds};
            let timerDisplay = document.getElementById('exam-timer');
            
            function updateTimer() {{
                if (timeRemaining <= 0) {{
                    // Time's up
                    alert('زمان امتحان به پایان رسیده است!');
                    document.getElementById('submit-exam-btn').click();
                    return;
                }}
                
                let minutes = Math.floor(timeRemaining / 60);
                let seconds = timeRemaining % 60;
                
                if (timerDisplay) {{
                    timerDisplay.textContent = 
                        String(minutes).padStart(2, '0') + ':' + 
                        String(seconds).padStart(2, '0');
                }}
                
                // Warning when 5 minutes remaining
                if (timeRemaining === 300) {{
                    alert('توجه: 5 دقیقه به پایان امتحان باقی مانده است!');
                }}
                
                timeRemaining--;
            }}
            
            // Start timer
            updateTimer();
            setInterval(updateTimer, 1000);
        }})();
        """