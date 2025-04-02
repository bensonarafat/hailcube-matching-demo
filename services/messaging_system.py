import smtplib
from slack_sdk import WebClient
from jinja2 import Template
import os
from dotenv import load_dotenv
from .logging_system import MessageLogger
from email.mime.text import MIMEText
from email.header import Header

load_dotenv()

class MessagingSystem:
    def __init__(self):
        self.logger = MessageLogger()
        self.email_template = Template(open('templates/email_template.j2').read())
        self.slack_template = Template(open('templates/slack_template.j2').read())
        
    def send_email(self, contact, opportunity):
        message = self.email_template.render(
            contact=contact,
            opportunity=opportunity
        ).encode('utf-8')
        
        msg = MIMEText(message, 'html', 'utf-8')
        msg['Subject'] = Header(f"New Opportunity: {opportunity['title']}", 'utf-8')
        msg['From'] = os.getenv('EMAIL_FROM')
        msg['To'] = contact['email']
        
        with smtplib.SMTP(os.getenv('SMTP_SERVER'), os.getenv('SMTP_PORT')) as server:
                server.starttls()
                server.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASS'))
                server.send_message(msg)
    
    def send_slack(self, contact, opportunity):
        client = WebClient(token=os.getenv('SLACK_TOKEN'))
        message = self.slack_template.render(
            contact=contact,
            opportunity=opportunity
        )
        
        client.chat_postMessage(
            channel=contact['slack_id'],
            text=message
        )
    
    def send_message(self, contact, opportunity):
        
        try:
            channel = contact['preferred_contact']
             
            if contact['preferred_contact'] == 'email':
                self.send_email(contact, opportunity)
            elif contact['preferred_contact'] == 'slack':
                self.send_slack(contact, opportunity)
                
            self.logger.log_message(
                contact, opportunity, 
                channel, 'success'
            )
        except Exception as e: 
            self.logger.log_message(
                contact, opportunity,
                channel, 'failed', e
            )
            raise
