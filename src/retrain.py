import os
import json
import yaml
import pymysql
import pandas as pd
import joblib
import mlflow
from dotenv import load_dotenv

# Ensure DLL loader runs first on Windows
try:
    import src.dll_loader
except ImportError:
    try:
        import dll_loader
    except ImportError:
        pass

try:
    from .data_preprocessing import TextPreprocessor
    from .model import build_model
except ImportError:
    from data_preprocessing import TextPreprocessor
    from model import build_model

load_dotenv()

def retrain_model():
    print("Starting model retraining loop...")
    
    # 1. Load config
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
        
    # 2. Connect to DB and fetch corrected tickets
    corrected_data = []
    try:
        conn = pymysql.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "1987"),
            database=os.getenv("DB_NAME", "customers_db"),
            cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cursor:
            cursor.execute("SELECT query_text as text, issue as label FROM issues WHERE is_corrected = TRUE")
            rows = cursor.fetchall()
            for r in rows:
                corrected_data.append({"text": r["text"], "label": r["label"]})
        conn.close()
        print(f"Found {len(corrected_data)} corrected tickets in database to append to training set.")
    except Exception as e:
        print(f"Warning: Failed to fetch corrections from database: {e}. Proceeding with base dataset.")

    # 3. Load base training data
    train_df = pd.read_csv(cfg["data"]["train_path"])
    
    # Append corrected data if any
    if corrected_data:
        extra_df = pd.DataFrame(corrected_data)
        train_df = pd.concat([train_df, extra_df], ignore_index=True)
        print(f"Total training dataset size after corrections: {len(train_df)}")

    # 4. Load validation and test data
    val_df = pd.read_csv(cfg["data"]["val_path"])
    test_df = pd.read_csv(cfg["data"]["test_path"])

    # 5. Fit preprocessor
    preprocessor = TextPreprocessor(
        max_features=cfg["preprocessing"]["max_features"],
        ngram_range=tuple(cfg["preprocessing"]["ngram_range"]),
        min_df=cfg["preprocessing"]["min_df"],
        max_df=cfg["preprocessing"]["max_df"],
        sublinear_tf=cfg["preprocessing"]["sublinear_tf"],
    )
    
    # Extract labels
    X_train_raw, y_train_raw = train_df["text"], train_df["label"]
    X_val_raw, y_val_raw = val_df["text"], val_df["label"]
    X_test_raw, y_test_raw = test_df["text"], test_df["label"]

    preprocessor.fit(X_train_raw, y_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    y_train = preprocessor.encode_labels(y_train_raw)
    y_val = preprocessor.encode_labels(y_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    y_test = preprocessor.encode_labels(y_test_raw)
    
    preprocessor.save("artifacts/preprocessor.pkl")
    print("Preprocessor fitted and saved.")

    # 6. Read active model from config
    active_model_name = cfg["models"]["active_model"]
    print(f"Training active model: {active_model_name}")

    # Load best hyperparameters from previous meta if available, else build default
    best_params = {}
    if os.path.exists("artifacts/best_model_meta.json"):
        try:
            with open("artifacts/best_model_meta.json") as f:
                meta = json.load(f)
                if meta.get("model_name") == active_model_name:
                    best_params = meta.get("best_params", {})
        except Exception:
            pass

    # Build model
    model = build_model(active_model_name, random_state=cfg["training"]["random_state"], **best_params)
    
    # Train
    is_seq = (active_model_name == "lstm")
    X_tr_in = X_train_raw if is_seq else X_train
    model.fit(X_tr_in, y_train)

    # Save best model
    joblib.dump(model, "artifacts/best_model.joblib")
    
    # Save meta
    # Calculate test accuracy
    X_te_in = X_test_raw if is_seq else X_test
    preds = model.predict(X_te_in)
    from sklearn.metrics import accuracy_score
    test_acc = accuracy_score(y_test, preds)
    print(f"Retraining complete. Test Accuracy: {test_acc:.4%}")

    # Log to MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    
    with mlflow.start_run(run_name=f"{active_model_name}_retrained"):
        mlflow.log_params({
            "model_name": active_model_name,
            "train_size": len(train_df),
            "added_corrections": len(corrected_data),
            **best_params
        })
        mlflow.log_metrics({
            "test_accuracy": float(test_acc)
        })
        mlflow.sklearn.log_model(model, artifact_path=f"model_{active_model_name}_retrained")
        
    print("Retrained model logged to MLflow.")
    return test_acc

if __name__ == "__main__":
    retrain_model()
