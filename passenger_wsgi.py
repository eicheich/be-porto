import sys
import os

# Menambahkan direktori root aplikasi ke sys.path agar modul lokal (src, model, dll.) dapat terbaca
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Phusion Passenger di cPanel membutuhkan variabel 'application' bertipe WSGI callable
from app import app as application

if __name__ == "__main__":
    application.run()

