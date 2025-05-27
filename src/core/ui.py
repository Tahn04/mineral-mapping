from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime

class ProgressBar:
    def __init__(self, total, desc="Processing"):
        self.pbar = tqdm(total=total, desc=desc)

    def update(self, n=1):
        self.pbar.update(n)

    def close(self):
        self.pbar.close()

class ProgressTable:
    def __init__(self):
        self.records = []

    def add_record(self, process_name, status, duration=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.records.append({
            "Time": timestamp,
            "Process": process_name,
            "Status": status,
            "Duration": duration if duration is not None else "-"
        })

    def show(self):
        if not self.records:
            print("No records to display.")
            return
        print(tabulate(self.records, headers="keys", tablefmt="grid"))

# Example usage:
pb = ProgressBar(10, desc="Downloading")
for i in range(10):
    pb.update()
pb.close()

pt = ProgressTable()
pt.add_record("Download", "Success", "5s")
pt.add_record("Upload", "Failed")
pt.show()