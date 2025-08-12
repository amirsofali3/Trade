"""
Thread-safe locks for atomic persistence operations
"""
import threading

# Global locks for atomic file operations
rfe_lock = threading.Lock()
version_lock = threading.Lock()