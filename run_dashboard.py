import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard.app import run_dashboard

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print("=" * 60)
    print("ZYNETRAX - Starting...")
    print(f"http://0.0.0.0:{port}")
    print("=" * 60)
    app = run_dashboard()
    app.run_server(debug=False, host="0.0.0.0", port=port)
