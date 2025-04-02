import smtplib
from slack_sdk import WebClient
from jinja2 import Template
import os
from dotenv import load_dotenv
from .logging_system import MessageLogger
from email.mime.text import MIMEText
from email.header import Header
import pandas as pd

load_dotenv()

class MessagingSystem:
    def __init__(self, contacts_file='data/contacts.csv'):
        self.logger = MessageLogger()
        self.email_template = Template(open('templates/email_template.j2').read())
        self.slack_template = Template(open('templates/slack_template.j2').read())
        self.contacts_file = contacts_file
        
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
        
    def update_contact_status(self, contact_id, status):
         """Update a contact's status in the CSV file"""
         try:
            # Read the current contacts file
            contacts_df = pd.read_csv(self.contacts_file)
            
            # Update the status for the specific contact
            contacts_df.loc[contacts_df['id'] == contact_id, 'status'] = status

            # Write the updated DataFrame back to the CSV file
            contacts_df.to_csv(self.contacts_file, index=False)

            print(f"Updated status for contact {contact_id} to {status}")
            
            return True
         except Exception as e:
            print(f"Error updating contact status: {str(e)}")
            return False
    
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
            
            # Update the contact's status to 0 (inactive) due to messaging failure
            contact_id = contact['id']
            self.update_contact_status(contact_id, 0)
            
            print(f"Message to contact {contact_id} failed - marked as inactive (status=0)")
            
            raise
