from datetime import datetime
import pytz

def show_timestamps():
    dt = datetime.now()
    print("Local:", dt)

    utc = datetime.now(pytz.utc)
    print("UTC:", utc)

    india = datetime.now(pytz.timezone("Asia/Kolkata"))
    print("India:", india)

if __name__ == "__main__":
    show_timestamps()
