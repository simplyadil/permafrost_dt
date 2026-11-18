"""
CR1000 Public Table Diagnostic
--------------------------------
Tests fetching the Public table to see what's currently
in it and whether the connection is stable enough.

Requirements:
    pip install pycampbellcr1000

Usage:
    python cr1000_public_test.py
"""

from pycampbellcr1000 import CR1000
from datetime import datetime
import threading

SERIAL_PORT     = "COM4"
BAUD_RATE       = 115200
CONNECT_TIMEOUT = 60
CALL_TIMEOUT    = 30

def call_with_timeout(fn, timeout, *args, **kwargs):
    result = [None]
    error  = [None]
    def target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            error[0] = e
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"Timed out after {timeout}s")
    if error[0]:
        raise error[0]
    return result[0]

def now():
    return datetime.now().strftime("%H:%M:%S")

def main():
    print(f"[{now()}] Connecting...")
    device = call_with_timeout(CR1000.from_url, CONNECT_TIMEOUT,
                               f"serial:{SERIAL_PORT}:{BAUD_RATE}")
    print(f"[{now()}] Connected!")

    # Step 1: list tables
    print(f"[{now()}] Listing tables...")
    tables = call_with_timeout(device.list_tables, CALL_TIMEOUT)
    print(f"[{now()}] Tables: {tables}")

    # Step 2: try fetching Public table data directly
    print(f"[{now()}] Fetching Public table...")
    data = call_with_timeout(device.get_data, CALL_TIMEOUT, "Public")
    if data:
        latest = data[-1]
        print(f"[{now()}] Public table has {len(latest)} fields:")
        for key, val in latest.items():
            print(f"  {key}: {val}")
    else:
        print(f"[{now()}] Public table returned no data.")

    device.bye()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")