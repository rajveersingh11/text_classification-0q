import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://raw.githubusercontent.com/bitext/customer-support-llm-chatbot-training-dataset/main/data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
DATA_DIR = "data"

INTENT_MAP = {
    # Payment Issue
    "payment_issue": "Payment Issue",
    "check_payment_methods": "Payment Issue",
    "check_invoice": "Payment Issue",
    "get_invoice": "Payment Issue",
    
    # Technical Problem
    "registration_problems": "Technical Problem",
    "recover_password": "Technical Problem",
    "complaint": "Technical Problem",
    
    # Product Inquiry
    "contact_customer_service": "Product Inquiry",
    "contact_human_agent": "Product Inquiry",
    "place_order": "Product Inquiry",
    "review": "Product Inquiry",
    "newsletter_subscription": "Product Inquiry",
    
    # Refund Request
    "get_refund": "Refund Request",
    "track_refund": "Refund Request",
    "check_refund_policy": "Refund Request",
    "cancel_order": "Refund Request",
    "check_cancellation_fee": "Refund Request",
    
    # Delivery Issue
    "track_order": "Delivery Issue",
    "delivery_options": "Delivery Issue",
    "delivery_period": "Delivery Issue",
    "change_shipping_address": "Delivery Issue",
    "set_up_shipping_address": "Delivery Issue",
    "change_order": "Delivery Issue"
}

def main():
    print(f"Downloading dataset from: {URL}")
    r = requests.get(URL)
    if r.status_code != 200:
        print(f"Error downloading dataset: HTTP {r.status_code}")
        return
        
    print("Download complete. Processing...")
    df = pd.read_csv(requests.compat.StringIO(r.text))
    
    # Filter and map
    df = df[df["intent"].isin(INTENT_MAP.keys())]
    df["label"] = df["intent"].map(INTENT_MAP)
    
    # Rename columns to match existing setup (text, label)
    df = df.rename(columns={"instruction": "text"})
    df = df[["text", "label"]]
    
    print("\nTarget label counts in new dataset:")
    print(df["label"].value_counts())
    
    # Split dataset
    train, temp = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp["label"], random_state=42
    )
    
    os.makedirs(DATA_DIR, exist_ok=True)
    train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    print(f"\nSaved new datasets under '{DATA_DIR}/':")
    print(f"  - train.csv: {len(train)} rows")
    print(f"  - val.csv: {len(val)} rows")
    print(f"  - test.csv: {len(test)} rows")

if __name__ == "__main__":
    main()
