# Hailcube Matching demo

This app handles sending real-time opportunities to contacts, integrated Celery with Flask to handle asynochronous tasks, using RabbitMQ as a message broker

## Prerequisites
- Python 3.13.0
- RabbitMQ
- pip package manager 

# Installation 

1. **Clone the repository**
   ```bash
   git clone https://github.com/bensonarafat/hailcube-matching-demo.git
   cd hailcube-matching-demo 

2. **Install RabbitMQ**
   ```bash 
   brew install rabbitmq
3. **Start RabbitMQ**
   ```bash
   rabbitmq-server 
   ```
4. **Install dependencies**
   ```bash 
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file, copy the content of `.env.example` and paste in the `.env` file 

```env
SMTP_SERVER=xxx
SMTP_PORT=587
EMAIL_FROM=xxx
SMTP_USER=xxx
SMTP_PASS=xx

SLACK_TOKEN=xxx

RABBITMQ_URL=amqp://guest:guest@localhost:5672//
RESULT_BACKEND=rpc://
```

Replace `xxx` with the correct values 

**Start Flask**

```bash 
flask run
```

**Start Celery**
```bash 
celery -A tasks.task_definitions worker --loglevel=info
```

Dashboard 
```
http://127.0.0.1:5000/
```

Trigger matching & opportunity 
```curl
http://127.0.0.1:5000/trigger-matching
```

Add new Opportunity
```curl
http://127.0.0.1:5000/add-opportunity
```

Message Logs
```curl 
http://127.0.0.1:5000/message-logs
```