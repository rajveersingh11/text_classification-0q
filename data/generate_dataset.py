"""
Synthetic dataset generator for support ticket classification.
Generates realistic-looking text samples for each category.
"""

import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split

# ────────────────────── TEMPLATE POOLS PER CLASS ──────────────────────

TEMPLATES = {
    "Payment Issue": [
        "My payment of ${amount} failed and I was charged twice on my card.",
        "I am unable to complete the payment for order #{order_id}. Please help.",
        "The payment gateway is showing an error during checkout.",
        "I noticed a duplicate charge on my credit card statement today.",
        "My payment method was declined but my card has sufficient funds.",
        "I want to update the payment method on my account urgently.",
        "There is an issue with the payment processing for my subscription.",
        "I was charged ${amount} but the order status shows unpaid.",
        "Payment is pending for 3 days and I have not received confirmation.",
        "Why is my payment not going through? I've tried multiple times.",
        "I need assistance with a failed transaction on my account.",
        "The autopay payment did not deduct this month and I got a late notice.",
        "I am seeing an unknown charge of ${amount} on my statement.",
        "Can you tell me why my payment was rejected? My bank says it's fine.",
        "I want to pay via a different method; the current one is broken.",
    ],
    "Technical Problem": [
        "The app keeps crashing every time I try to open it on my phone.",
        "I cannot log into my account; the login button does nothing.",
        "The website is not loading properly; I get a 500 error message.",
        "The mobile app is extremely slow and freezes frequently.",
        "I am getting a 'Session Expired' error constantly on the site.",
        "The search functionality returns no results even with valid queries.",
        "I cannot upload any images to the platform; the upload fails.",
        "The system is throwing a 404 error on every page I visit.",
        "My dashboard is showing a blank white screen after login.",
        "There is a bug in the checkout flow; it loops back to cart.",
        "The notifications are not coming through on my device anymore.",
        "I am unable to download the invoice PDF; it gives an error.",
        "Two-factor authentication is not sending the OTP code to my phone.",
        "The integration with Slack is broken and not syncing messages.",
        "The mobile app crashes immediately after the splash screen.",
    ],
    "Product Inquiry": [
        "Does the premium plan include advanced analytics features?",
        "I would like to know more about the specifications of product X.",
        "Is there a free trial available for the enterprise edition?",
        "What are the differences between the basic and pro subscriptions?",
        "Can you tell me which plan supports multi-user collaboration?",
        "Is this product compatible with Mac OS Ventura?",
        "Do you offer discounts for annual billing on the starter plan?",
        "What languages does the platform support for international users?",
        "How many devices can I use with a single account license?",
        "Are there any upcoming features in your product roadmap?",
        "I want to know if your software integrates with Salesforce.",
        "Could you share the system requirements for installation?",
        "What security measures are in place to protect my data?",
        "Do you have a mobile version of the desktop application?",
        "Is training or onboarding included with the enterprise purchase?",
    ],
    "Refund Request": [
        "I want to request a refund for order #{order_id} placed last week.",
        "Please process my refund of ${amount} as I am not satisfied.",
        "I have cancelled my subscription and would like a full refund.",
        "The item arrived damaged; I would like to initiate a refund.",
        "I was overcharged and would like the difference refunded.",
        "How long does it take to receive a refund after cancellation?",
        "I never received my order; please issue a refund immediately.",
        "I would like to return the product and get my money back.",
        "Can you confirm whether my refund of ${amount} has been processed?",
        "The service did not match the description; I want a refund.",
        "I have been waiting 2 weeks for my refund. Please look into it.",
        "Please cancel my order and refund the amount to my card.",
        "I was double-billed and need a refund for the extra charge.",
        "I want a refund because the trial version did not meet my needs.",
        "I changed my mind about the purchase; please issue a refund.",
    ],
    "Delivery Issue": [
        "My order #{order_id} was supposed to arrive yesterday but hasn't.",
        "The package was marked as delivered but I never received it.",
        "I have been waiting 10 days for delivery with no update.",
        "The tracking number shows my order is stuck in transit.",
        "My order arrived with missing items; some products are absent.",
        "The delivery was delayed and I need it urgently for an event.",
        "I received the wrong item in my delivery; it is not what I ordered.",
        "The package was left at the wrong address; how can I get it back?",
        "Can you provide an updated delivery estimate for my shipment?",
        "I want to change the delivery address for an in-transit order.",
        "The courier attempted delivery but I was not home; please redeliver.",
        "My order shows shipped but no tracking info is available.",
        "The package arrived damaged; the box was crushed and items broken.",
        "I need to reschedule the delivery date for my upcoming order.",
        "Where is my order? It has been 2 weeks and still says processing.",
    ],
}

# Augmentation helpers
SLOT_FILLERS = {
    "{amount}": ["49.99", "129.00", "75.50", "19.99", "299.00", "59.99"],
    "{order_id}": ["12345", "78901", "45213", "98765", "33442", "55667"],
}

PREFIXES = [
    "Hi team,", "Hello,", "Good morning,", "Hi,", "Dear support,",
    "Hey there,", "Greetings,", "", ""
]

SUFFIXES = [
    "Please help me resolve this as soon as possible.",
    "Thanks for your support.",
    "I appreciate your quick response.",
    "Looking forward to hearing from you.",
    "This is urgent, please respond quickly.",
    "Kindly assist at the earliest.",
    "Thank you.",
    "I have attached the relevant screenshots for reference.",
    "Please escalate this issue if needed.",
    "",
    "",
]


def augment_text(text: str) -> str:
    """Fill in placeholders and optionally add prefix/suffix."""
    for slot, fillers in SLOT_FILLERS.items():
        text = text.replace(slot, random.choice(fillers))
    return text


def generate_samples(per_class: int = 1500) -> pd.DataFrame:
    rows = []
    for label, templates in TEMPLATES.items():
        for _ in range(per_class):
            base = random.choice(templates)
            text = augment_text(base)
            if random.random() < 0.3:
                text = random.choice(PREFIXES) + " " + text
            if random.random() < 0.5:
                text = text + " " + random.choice(SUFFIXES)
            rows.append({"text": text.strip(), "label": label})
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def split_and_save(df: pd.DataFrame, out_dir: str = "data"):
    os.makedirs(out_dir, exist_ok=True)
    train, temp = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp["label"], random_state=42
    )
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"✅ Saved: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    df = generate_samples(per_class=1500)
    print(f"Generated {len(df)} samples")
    print(df["label"].value_counts())
    split_and_save(df)
