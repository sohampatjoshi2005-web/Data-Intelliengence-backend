-- Local PostgreSQL demo dataset for Connectors testing.
-- Run: psql -d postgres -f sample_datasets/postgres/seed_connector_demo.sql

CREATE DATABASE connector_demo;

\connect connector_demo

CREATE TABLE IF NOT EXISTS customer_churn (
  customer_id SERIAL PRIMARY KEY,
  age INTEGER NOT NULL,
  tenure_months INTEGER NOT NULL,
  monthly_charges NUMERIC(10,2) NOT NULL,
  total_charges NUMERIC(10,2) NOT NULL,
  contract_type VARCHAR(20) NOT NULL,
  internet_service VARCHAR(20) NOT NULL,
  tech_support VARCHAR(20) NOT NULL,
  churn INTEGER NOT NULL
);

TRUNCATE customer_churn RESTART IDENTITY;

INSERT INTO customer_churn (
  age, tenure_months, monthly_charges, total_charges,
  contract_type, internet_service, tech_support, churn
) VALUES
(25, 1, 29.85, 29.85, 'Month-to-month', 'DSL', 'No', 1),
(30, 34, 56.95, 1889.50, 'One year', 'Fiber', 'Yes', 0),
(45, 2, 53.85, 108.15, 'Month-to-month', 'DSL', 'No', 1),
(37, 45, 42.30, 1840.75, 'Two year', 'None', 'No internet', 0),
(52, 10, 70.35, 675.20, 'Month-to-month', 'Fiber', 'No', 1),
(29, 24, 45.00, 1080.00, 'One year', 'DSL', 'Yes', 0),
(41, 6, 65.60, 393.60, 'Month-to-month', 'Fiber', 'No', 1),
(33, 18, 39.40, 709.20, 'One year', 'DSL', 'Yes', 0),
(48, 3, 89.10, 267.30, 'Month-to-month', 'Fiber', 'No', 1),
(27, 36, 55.20, 1987.20, 'Two year', 'Fiber', 'Yes', 0),
(39, 8, 49.99, 399.92, 'Month-to-month', 'DSL', 'No', 1),
(44, 52, 61.75, 3211.00, 'Two year', 'Fiber', 'Yes', 0),
(31, 5, 33.25, 166.25, 'Month-to-month', 'None', 'No internet', 0),
(36, 15, 58.40, 876.00, 'One year', 'DSL', 'Yes', 0),
(50, 12, 74.80, 897.60, 'Month-to-month', 'Fiber', 'No', 1),
(28, 28, 47.15, 1320.20, 'One year', 'DSL', 'Yes', 0),
(42, 4, 62.00, 248.00, 'Month-to-month', 'Fiber', 'No', 1),
(35, 40, 51.30, 2052.00, 'Two year', 'DSL', 'Yes', 0),
(46, 7, 68.90, 482.30, 'Month-to-month', 'Fiber', 'No', 1),
(32, 22, 44.55, 980.10, 'One year', 'DSL', 'Yes', 0);

DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'connector_user') THEN
    CREATE ROLE connector_user WITH LOGIN PASSWORD 'connector_pass';
  ELSE
    ALTER ROLE connector_user WITH LOGIN PASSWORD 'connector_pass';
  END IF;
END $$;

GRANT ALL PRIVILEGES ON DATABASE connector_demo TO connector_user;
GRANT ALL ON SCHEMA public TO connector_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO connector_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO connector_user;
