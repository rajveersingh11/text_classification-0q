import os
import time
import requests

EMAIL_LOG_PATH = "artifacts/email_alerts.log"
SLACK_LOG_PATH = "artifacts/slack_alerts.log"

def send_alerts(issue_id: int, name: str, category: str, text: str, priority: str, department: str):
    """
    Send alerts via Mock Email and Mock/Real Slack Webhook for Critical issues.
    """
    os.makedirs("artifacts", exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # 1. Dispatch Email Alert
    email_body = f"""==================================================
EMAIL ALERT - [{timestamp}]
To: {department.lower().replace('&', 'and').replace(' ', '_')}@company.com
Subject: URGENT: Critical Priority Ticket #{issue_id} Ingested
--------------------------------------------------
Ticket ID: #{issue_id}
Customer Name: {name}
Inferred Category: {category}
Assigned Department: {department}
Priority: {priority}

Customer Ticket Query:
"{text}"

Please resolve this ticket immediately.
==================================================
"""
    with open(EMAIL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(email_body + "\n")
    print(f"Mock email alert appended to {EMAIL_LOG_PATH}")

    # 2. Dispatch Slack Webhook Alert
    slack_payload = {
        "text": f"🚨 *CRITICAL TICKET INGESTED* 🚨",
        "attachments": [
            {
                "color": "#ef4444",
                "fields": [
                    {"title": "Ticket ID", "value": f"#{issue_id}", "short": True},
                    {"title": "Customer", "value": name, "short": True},
                    {"title": "Routed Department", "value": department, "short": True},
                    {"title": "Category", "value": category, "short": True},
                    {"title": "Priority", "value": priority, "short": True},
                ],
                "text": f"*Query:* _\"{text}\"_"
            }
        ]
    }

    # Log mock Slack webhook
    import json
    with open(SLACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] payload: {json.dumps(slack_payload)}\n")
    print(f"Mock Slack alert payload logged to {SLACK_LOG_PATH}")

    # Try sending to actual Slack Webhook if configured
    slack_url = os.getenv("SLACK_WEBHOOK_URL")
    if slack_url:
        try:
            r = requests.post(slack_url, json=slack_payload, timeout=5)
            print(f"Slack webhook status: {r.status_code}")
        except Exception as e:
            print(f"Failed to post to real Slack webhook: {e}")
