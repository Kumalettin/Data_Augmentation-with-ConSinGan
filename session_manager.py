"""
Session management for user/session-based file organization.
"""

import os
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages user sessions with isolated file directories.
    Handles session creation, path management, and cleanup of old sessions.
    """

    UPLOADS_DIR = "uploads"
    MODELS_DIR = "models"
    LOGS_DIR = "logs"

    def __init__(self):
        """Initialize session manager and create necessary directories."""
        self._ensure_directories()

    @staticmethod
    def _ensure_directories():
        """Ensure all required directories exist."""
        for directory in [SessionManager.UPLOADS_DIR, SessionManager.MODELS_DIR, SessionManager.LOGS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def create_session():
        """
        Create a new session with a unique ID.
        
        Returns:
            str: Unique session ID (UUID format)
        """
        session_id = str(uuid.uuid4())
        
        # Create session directories
        uploads_path = os.path.join(SessionManager.UPLOADS_DIR, session_id)
        models_path = os.path.join(SessionManager.MODELS_DIR, session_id)
        
        Path(uploads_path).mkdir(parents=True, exist_ok=True)
        Path(models_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Session created: {session_id}")
        return session_id

    @staticmethod
    def get_session_path(session_id, path_type="uploads"):
        """
        Get the full path for a session directory.
        
        Args:
            session_id (str): Session ID
            path_type (str): Type of path ("uploads", "models", or "logs")
        
        Returns:
            str: Full path to the session directory
        
        Raises:
            ValueError: If path_type is invalid
        """
        if path_type == "uploads":
            base_dir = SessionManager.UPLOADS_DIR
        elif path_type == "models":
            base_dir = SessionManager.MODELS_DIR
        elif path_type == "logs":
            base_dir = SessionManager.LOGS_DIR
        else:
            raise ValueError(f"Invalid path_type: {path_type}")
        
        return os.path.join(base_dir, session_id)

    @staticmethod
    def cleanup_old_sessions(max_age_hours=24):
        """
        Clean up sessions older than max_age_hours.
        Deletes both uploads and models directories for old sessions.
        
        Args:
            max_age_hours (int): Maximum age of sessions in hours (default: 24)
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for directory in [SessionManager.UPLOADS_DIR, SessionManager.MODELS_DIR]:
            if not os.path.exists(directory):
                continue
            
            for session_id in os.listdir(directory):
                session_path = os.path.join(directory, session_id)
                
                if not os.path.isdir(session_path):
                    continue
                
                # Get directory modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(session_path))
                
                if mod_time < cutoff_time:
                    try:
                        # Remove directory and all contents
                        import shutil
                        shutil.rmtree(session_path)
                        logger.info(f"Cleaned up old session: {session_id}")
                    except Exception as e:
                        logger.error(f"Failed to cleanup session {session_id}: {e}")

    @staticmethod
    def get_active_sessions():
        """
        Get list of all active session IDs.
        
        Returns:
            list: List of active session IDs
        """
        active_sessions = set()
        
        for directory in [SessionManager.UPLOADS_DIR, SessionManager.MODELS_DIR]:
            if not os.path.exists(directory):
                continue
            
            for session_id in os.listdir(directory):
                session_path = os.path.join(directory, session_id)
                if os.path.isdir(session_path):
                    active_sessions.add(session_id)
        
        return list(active_sessions)

    @staticmethod
    def session_exists(session_id):
        """
        Check if a session exists.
        
        Args:
            session_id (str): Session ID to check
        
        Returns:
            bool: True if session exists, False otherwise
        """
        uploads_path = SessionManager.get_session_path(session_id, "uploads")
        models_path = SessionManager.get_session_path(session_id, "models")
        
        return os.path.exists(uploads_path) or os.path.exists(models_path)
