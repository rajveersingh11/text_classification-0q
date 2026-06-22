try:
    import src.dll_loader
except ImportError:
    pass

import os
import time
import pymysql
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from src.predict import TicketClassifier

try:
    from src.notifier import send_alerts
except ImportError:
    from notifier import send_alerts

try:
    from src.retrain import retrain_model
except ImportError:
    from retrain import retrain_model

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Ticket Classification & Support API",
    description="Real-time ML-based classification of customer support tickets with DB logging.",
    version="2.0.0",
)

MODEL_PATH = "artifacts/best_model.joblib"
PREP_PATH = "artifacts/preprocessor.pkl"
clf = TicketClassifier(MODEL_PATH, PREP_PATH)


# Persistent Database Connection
_db_conn = None

def get_db_connection():
    global _db_conn
    try:
        if _db_conn is not None:
            _db_conn.ping(reconnect=True)
            return _db_conn
    except Exception:
        _db_conn = None

    try:
        _db_conn = pymysql.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "1987"),
            database=os.getenv("DB_NAME", "customers_db"),
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True
        )
        return _db_conn
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}"
        )


class Ticket(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class BatchTickets(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500)


class Prediction(BaseModel):
    text: str
    prediction: str
    confidence: float
    top_k: list


class IssueCreate(BaseModel):
    customer_id: Optional[int] = None
    query_text: str = Field(..., min_length=1, max_length=5000)


class CustomerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    phone: Optional[str] = Field(None, max_length=50)
    email: Optional[str] = Field(None, max_length=255)
    account_number: Optional[str] = Field(None, max_length=100)
    card_number: Optional[str] = Field(None, max_length=100)
    aadhaar_number: Optional[str] = Field(None, max_length=100)
    pan_number: Optional[str] = Field(None, max_length=50)
    ifsc_code: Optional[str] = Field(None, max_length=50)
    pincode: Optional[str] = Field(None, max_length=20)



@app.get("/health")
def health():
    db_ok = True
    try:
        conn = get_db_connection()
    except Exception:
        db_ok = False
    return {
        "status": "ok",
        "model": type(clf.model).__name__,
        "database_connected": db_ok
    }


@app.get("/classes")
def classes():
    return {"classes": clf.classes_}


@app.post("/predict", response_model=Prediction)
def predict_single(ticket: Ticket):
    t0 = time.time()
    try:
        result = clf.predict([ticket.text], top_k=3)[0]
        result["latency_ms"] = round((time.time() - t0) * 1000, 2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[Prediction])
def predict_batch(batch: BatchTickets):
    try:
        return clf.predict(batch.texts, top_k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── DATABASE BACKED API ENDPOINTS ───

@app.get("/api/customers")
def get_customers(search: Optional[str] = None, limit: int = 50):
    """Retrieve list of customers from DB, optionally filtered by name."""
    conn = get_db_connection()
    with conn.cursor() as cursor:
        if search:
            query = "SELECT id, name, email, phone FROM customers WHERE name LIKE %s LIMIT %s"
            cursor.execute(query, (f"%{search}%", limit))
        else:
            query = "SELECT id, name, email, phone FROM customers LIMIT %s"
            cursor.execute(query, limit)
        return cursor.fetchall()


@app.post("/api/customers")
def create_customer(cust_in: CustomerCreate):
    """Add a new customer to the database, auto-incrementing ID."""
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM customers")
        next_id = cursor.fetchone()["next_id"]
        
        insert_query = """
            INSERT INTO customers (
                id, name, phone, email, account_number, card_number, 
                aadhaar_number, pan_number, ifsc_code, pincode
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            next_id,
            cust_in.name,
            cust_in.phone,
            cust_in.email,
            cust_in.account_number,
            cust_in.card_number,
            cust_in.aadhaar_number,
            cust_in.pan_number,
            cust_in.ifsc_code,
            cust_in.pincode
        ))
        
    return {
        "id": next_id,
        "name": cust_in.name,
        "email": cust_in.email,
        "phone": cust_in.phone,
        "status": "created_successfully"
    }


@app.get("/api/issues")
def get_issues(limit: int = 50):
    """Retrieve logged issues from DB."""
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT id, customer_id, name, issue, query_text, confidence, status, priority, assigned_department, created_at "
            "FROM issues ORDER BY id DESC LIMIT %s",
            limit
        )
        return cursor.fetchall()


@app.post("/api/issues")
def create_issue(issue_in: IssueCreate):
    """Classify a query, resolve customer name, and log to the database with routing metadata."""
    conn = get_db_connection()
    
    # Resolve customer name
    customer_name = "Anonymous"
    if issue_in.customer_id:
        with conn.cursor() as cursor:
            cursor.execute("SELECT name FROM customers WHERE id = %s", (issue_in.customer_id,))
            cust = cursor.fetchone()
            if cust:
                customer_name = cust["name"]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Customer with ID {issue_in.customer_id} does not exist."
                )

    # Run classification
    pred_res = clf.predict([issue_in.query_text], top_k=1)[0]
    predicted_label = pred_res["prediction"]
    confidence = pred_res["confidence"]

    # Ingest routing department mapping
    dept_map = {
        "Payment Issue": "Finance & Billing",
        "Refund Request": "Finance & Billing",
        "Technical Problem": "IT Support",
        "Delivery Issue": "Operations & Logistics",
        "Product Inquiry": "Sales & Product"
    }
    assigned_department = dept_map.get(predicted_label, "General Support")

    # Ingest priority heuristic
    text_lower = issue_in.query_text.lower()
    urgent_keywords = ["urgent", "immediately", "broke", "crash", "stolen", "lost", "charge twice", "double charge", "asap"]
    is_urgent = any(k in text_lower for k in urgent_keywords)

    if predicted_label in ["Payment Issue", "Refund Request"]:
        priority = "High" if not is_urgent else "Critical"
    elif predicted_label == "Technical Problem":
        priority = "High" if is_urgent else "Medium"
    else:
        priority = "Medium" if is_urgent else "Low"

    # Set status as 'Open'
    status = "Open"

    # Insert issue record
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO issues (customer_id, name, issue, query_text, confidence, status, priority, assigned_department) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (issue_in.customer_id, customer_name, predicted_label, issue_in.query_text, confidence, status, priority, assigned_department)
        )
        issue_id = cursor.lastrowid

    # If priority is Critical, dispatch alerts
    if priority == "Critical":
        try:
            send_alerts(issue_id, customer_name, predicted_label, issue_in.query_text, priority, assigned_department)
        except Exception as alert_err:
            print(f"Failed to dispatch critical alerts: {alert_err}")

    return {
        "id": issue_id,
        "customer_id": issue_in.customer_id,
        "name": customer_name,
        "issue": predicted_label,
        "confidence": confidence,
        "status": status,
        "priority": priority,
        "assigned_department": assigned_department,
        "query_text": issue_in.query_text,
        "status_code": "logged_successfully"
    }


class IssueUpdate(BaseModel):
    issue: str = Field(..., min_length=1, max_length=100)


@app.put("/api/issues/{issue_id}")
def update_issue(issue_id: int, issue_update: IssueUpdate):
    """Manually correct routing category and update department, priority, and mark as corrected."""
    conn = get_db_connection()
    new_category = issue_update.issue
    
    # Verify new category is valid
    if new_category not in clf.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{new_category}'. Allowed: {clf.classes_}"
        )

    # Ingest routing department mapping
    dept_map = {
        "Payment Issue": "Finance & Billing",
        "Refund Request": "Finance & Billing",
        "Technical Problem": "IT Support",
        "Delivery Issue": "Operations & Logistics",
        "Product Inquiry": "Sales & Product"
    }
    assigned_department = dept_map.get(new_category, "General Support")

    # Fetch query text from DB
    with conn.cursor() as cursor:
        cursor.execute("SELECT query_text, issue, original_issue, is_corrected FROM issues WHERE id = %s", (issue_id,))
        issue_record = cursor.fetchone()
        if not issue_record:
            raise HTTPException(
                status_code=404,
                detail=f"Issue with ID {issue_id} not found."
            )
        
        query_text = issue_record["query_text"]
        current_issue = issue_record["issue"]
        orig_issue = issue_record["original_issue"] if issue_record["original_issue"] else current_issue
        
    text_lower = query_text.lower()
    urgent_keywords = ["urgent", "immediately", "broke", "crash", "stolen", "lost", "charge twice", "double charge", "asap"]
    is_urgent = any(k in text_lower for k in urgent_keywords)

    if new_category in ["Payment Issue", "Refund Request"]:
        priority = "High" if not is_urgent else "Critical"
    elif new_category == "Technical Problem":
        priority = "High" if is_urgent else "Medium"
    else:
        priority = "Medium" if is_urgent else "Low"

    # Update database record
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE issues SET "
            "original_issue = %s, "
            "issue = %s, "
            "assigned_department = %s, "
            "priority = %s, "
            "is_corrected = TRUE "
            "WHERE id = %s",
            (orig_issue, new_category, assigned_department, priority, issue_id)
        )

    return {
        "id": issue_id,
        "original_issue": orig_issue,
        "new_issue": new_category,
        "priority": priority,
        "assigned_department": assigned_department,
        "status": "updated_successfully"
    }


@app.post("/api/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger background model retraining on current dataset + corrections."""
    background_tasks.add_task(retrain_model)
    return {"status": "retraining_started", "message": "Model retraining has been queued in the background."}


# Mount static folder
os.makedirs("api/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="api/static"), name="static")


@app.get("/")
def read_index():
    return FileResponse("api/static/index.html")
