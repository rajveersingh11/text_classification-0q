CREATE DATABASE IF NOT EXISTS customers_db;
USE customers_db;

CREATE TABLE IF NOT EXISTS issues (
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

SELECT 'issues table created successfully' AS status;
