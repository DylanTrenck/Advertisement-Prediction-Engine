# Gunicorn configuration file
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '80')}"

# Worker processes
workers = int(os.getenv("WEB_CONCURRENCY", "2"))
worker_class = "sync"

# Timeout
timeout = 120  # 2 minutes

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "audio_features_app"

# Server mechanics
daemon = False
pidfile = None

