from flask import Flask, request, redirect, url_for, render_template
import pandas as pd
from config import Config
from services.celery_service import init_celery

# Import tasks after celery is initialized
# This avoids circular imports
from tasks.task_definitions import run_matching

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Celery with Flask app
celery = init_celery(app)

@app.route('/')
def dashboard():
    contacts = pd.read_csv('data/contacts.csv') 
    opportunities = pd.read_csv('data/opportunities.csv')
    return render_template('index.html',
                           contacts=dataframe_to_html(contacts), 
                           opportunities=dataframe_to_html(opportunities))

@app.route('/trigger-matching')
def trigger_matching():
    run_matching.delay()
    return render_template('index.html', 
                           status="Matching process started!",
                           contacts=dataframe_to_html(pd.read_csv('data/contacts.csv')),
                           opportunities=dataframe_to_html(pd.read_csv('data/opportunities.csv')))

@app.route('/add-opportunity', methods=['GET', 'POST'])
def add_opportunity():
    if request.method == 'POST':
        try:
            opp_df = pd.read_csv('data/opportunities.csv')
             # Generate new ID (next number in sequence)
            if len(opp_df) == 0:
                new_id = 1
            else:
                new_id = int(opp_df['id'].max()) + 1
            # Get form data 
            new_opp = {
                'id': new_id,
                'opportunity_type': request.form['type'],
                'title': request.form['title'],
                'description': request.form['description'],
                'required_skills': request.form['skills'],
                'urgency': request.form['urgency']
            }
            # Save to CSV

            opp_df = pd.concat([opp_df, pd.DataFrame([new_opp])], ignore_index=True) 
            opp_df.to_csv('data/opportunities.csv', index=False)
            
            # Trigger matching immediately
            run_matching.delay(opportunity_id=new_id)
            
            return redirect(url_for('dashboard'))
        except Exception as e:
            return redirect(url_for('dashboard'))
        
    return render_template('add_opportunity.html')


@app.route('/message-logs')
def view_logs():
    try:
        logs_df = pd.read_csv("logs/messaging_logs.csv")
        return render_template('logs.html', logs=dataframe_to_html(logs_df))
    except FileNotFoundError:
        return render_template('logs.html', logs="No logs found")
    

@app.template_filter('format_log_timestamp')
def format_log_timestamp(value):
    return pd.to_datetime(value).strftime('%Y-%m-%d %H:%M:%S')


def dataframe_to_html(df):
    return df.to_html(classes='table table-striped table-bordered table-hover', 
                     index=False, 
                     escape=False,
                     border=0)

if __name__ == '__main__':
    app.run(debug=True)