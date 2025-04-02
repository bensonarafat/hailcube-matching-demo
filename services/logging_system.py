import csv
import os
from datetime import datetime
import pandas as pd

class MessageLogger:
    def __init__(self):
        self.log_dir = "logs"
        self.log_path = os.path.join(self.log_dir, "messaging_logs.csv")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.columns = [
            'timestamp', 'contact_id', 'opportunity_id',
            'channel', 'status', 'error', 'message_preview'
        ]
    
    def _create_initial_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)
    
    def log_message(self, contact, opportunity, channel, status, error=None):
        self._create_initial_log()
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'contact_id': contact['id'],
            'opportunity_id': opportunity['id'],
            'channel': channel,
            'status': status,
            'error': str(error)[:255] if error else None,
            'message_preview': f"{opportunity['title'][:50]}..."  # Truncate preview
        }
        
        # Append to CSV
        pd.DataFrame([log_entry]).to_csv(
            self.log_path,
            mode='a',
            header=False,
            index=False
        )