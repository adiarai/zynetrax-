import sys 
import os 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) 
from dashboard.app import run_server 
if __name__ == "__main__": 
    print("=" * 60) 
    print("ZYNETRAX - Starting...") 
    print("http://127.0.0.1:8050") 
    print("=" * 60) 
    run_server() 
