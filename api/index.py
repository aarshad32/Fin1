from http.server import BaseHTTPRequestHandler
from subprocess import Popen, PIPE
import os
import sys

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        
        # Get the current working directory
        cwd = os.getcwd()
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go back one directory to get to the project root
        project_dir = os.path.abspath(os.path.join(script_dir, ".."))
        
        try:
            # Change to the project root directory
            os.chdir(project_dir)
            
            # Run Streamlit app
            message = "This endpoint will redirect to a Streamlit app. Please deploy using Streamlit Cloud or another provider that supports long-running processes."
            self.wfile.write(message.encode())
            
        except Exception as e:
            self.wfile.write(str(e).encode())
        finally:
            # Change back to the original directory
            os.chdir(cwd)
        return