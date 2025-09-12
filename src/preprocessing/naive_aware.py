from datetime import datetime, timezone

def show_naive_aware():
    naive = datetime.now()
    aware = datetime.now(timezone.utc)
    print("Naive:", naive)
    print("Aware (UTC):", aware)

if __name__ == "__main__":
    show_naive_aware()
