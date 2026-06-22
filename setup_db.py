import pymysql
import pandas as pd
import sys
import os

host = "127.0.0.1"
port = 3306
password = "1987"
users = ["root", "mysql"]

conn = None
connected_user = None

# 1. Connect to MySQL server
for user in users:
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            autocommit=True
        )
        connected_user = user
        print(f"Successfully connected to MySQL as '{user}'")
        break
    except Exception as e:
        print(f"Failed to connect as '{user}': {e}")

if not conn:
    print("Could not connect to MySQL with any user. Please make sure MySQL is running on 127.0.0.1:3306.")
    sys.exit(1)

try:
    with conn.cursor() as cursor:
        # 2. Re-create database and tables
        cursor.execute("CREATE DATABASE IF NOT EXISTS customers_db;")
        cursor.execute("USE customers_db;")
        
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        cursor.execute("DROP TABLE IF EXISTS issues;")
        cursor.execute("""
            CREATE TABLE issues (
                id                  INT AUTO_INCREMENT PRIMARY KEY,
                customer_id         INT           NULL,
                name                VARCHAR(255)  NOT NULL,
                issue               VARCHAR(100)  NOT NULL,
                query_text          TEXT          NOT NULL,
                confidence          DOUBLE        NULL,
                status              VARCHAR(50)   NOT NULL DEFAULT 'Open',
                priority            VARCHAR(50)   NOT NULL DEFAULT 'Medium',
                assigned_department VARCHAR(100)  NULL,
                original_issue      VARCHAR(100)  NULL,
                is_corrected        BOOLEAN       NOT NULL DEFAULT FALSE,
                created_at          DATETIME      DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_issues_customer_id (customer_id),
                INDEX idx_issues_status (status)
            );
        """)
        print("Table 'issues' created.")

        cursor.execute("DROP TABLE IF EXISTS customers;")
        cursor.execute("""
            CREATE TABLE customers (
                id             INT PRIMARY KEY,
                name           VARCHAR(255)  NOT NULL,
                phone          VARCHAR(50)   NULL,
                email          VARCHAR(255)  NULL,
                account_number VARCHAR(100)  NULL,
                card_number    VARCHAR(100)  NULL,
                aadhaar_number VARCHAR(100)  NULL,
                pan_number     VARCHAR(50)   NULL,
                ifsc_code      VARCHAR(50)   NULL,
                pincode        VARCHAR(20)   NULL,
                INDEX idx_customers_name (name),
                INDEX idx_customers_email (email)
            );
        """)
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
        print("Table 'customers' created.")
        
        # 3. Read CSV data and insert into database table
        csv_path = os.path.join(os.path.dirname(__file__), "customers_db.csv")
        if not os.path.exists(csv_path):
            print(f"Error: csv file not found at {csv_path}")
            sys.exit(1)
            
        df = pd.read_csv(csv_path)
        df = df.where(pd.notnull(df), None)
        
        insert_query = """
            INSERT INTO customers (
                id, name, phone, email, account_number, card_number, 
                aadhaar_number, pan_number, ifsc_code, pincode
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for _, row in df.iterrows():
            cursor.execute(insert_query, (
                int(row["id"]),
                row["name"],
                row["phone"],
                row["email"],
                row["account_number"],
                row["card_number"],
                row["aadhaar_number"],
                row["pan_number"],
                row["ifsc_code"],
                row["pincode"]
            ))
            
        print(f"Successfully imported {len(df)} customer records into 'customers' table.")
        
        # 4. Link tables via Foreign Key
        try:
            cursor.execute("""
                ALTER TABLE issues
                ADD CONSTRAINT fk_issues_customer
                FOREIGN KEY (customer_id) REFERENCES customers(id)
                ON DELETE SET NULL;
            """)
            print("Foreign key constraint added to issues table.")
        except Exception as fk_err:
            print(f"Note: Foreign key constraint not added (it might already exist): {fk_err}")

        # 5. Write the environment variables configuration
        env_content = f"""DB_HOST={host}
DB_PORT={port}
DB_USER={connected_user}
DB_PASSWORD={password}
DB_NAME=customers_db
"""
        with open(".env", "w") as env_file:
            env_file.write(env_content)
        print(".env file configured successfully.")

except Exception as e:
    print(f"Error occurred during database configuration: {e}")
    sys.exit(1)
finally:
    conn.close()
