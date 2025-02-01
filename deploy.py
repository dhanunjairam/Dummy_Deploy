import smtplib
from email.mime.text import MIMEText
from fastapi import FastAPI, Query

app = FastAPI()

import os
import smtplib
from email.mime.text import MIMEText

smtp_server = os.getenv("SMTP_SERVER")
smtp_port = int(os.getenv("SMTP_PORT", "587"))
email_user = os.getenv("EMAIL_USER")
email_password = os.getenv("EMAIL_PASSWORD")

@app.get("/send-email/")
async def send_email(to_email: str = Query(..., description="Recipient's email address")):
    """
    Send an email to the specified recipient.
    The recipient email is passed as a query parameter.
    """
    msg = MIMEText('This is the body of the email')
    msg['Subject'] = 'Test Email'
    msg['From'] = email_user
    msg['To'] = to_email

    try:
        # Send email using SMTP server
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(email_user, email_password)
            server.sendmail(msg['From'], msg['To'], msg.as_string())
            return {"message": f"Email sent successfully to {to_email}!"}
    except Exception as e:
        return {"error": str(e)}
