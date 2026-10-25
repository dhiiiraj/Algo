import logging
import os
import smtplib
from email.mime.text import MIMEText

log = logging.getLogger("algo.notifications")


def send_email_report(subject: str, body: str) -> None:
    gmail_address = os.getenv("GMAIL_ADDRESS")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")
    email_to = os.getenv("EMAIL_TO", gmail_address)

    if not gmail_address or not gmail_app_password:
        log.warning(
            "GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set; skipping email report."
        )
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = email_to

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(gmail_address, gmail_app_password)
            server.sendmail(
                gmail_address,
                [a.strip() for a in email_to.split(",")],
                msg.as_string(),
            )
        log.info("Email report sent to %s", email_to)
    except Exception as e:
        log.warning("Failed to send email report: %s", e)
