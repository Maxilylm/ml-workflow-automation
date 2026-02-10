-- Snowflake Database and Schema Setup for ML Project
-- Run this script to initialize the Snowflake environment

-- =============================================================================
-- DATABASE AND SCHEMAS
-- =============================================================================

-- Create main database
CREATE DATABASE IF NOT EXISTS ML_PROJECT;

USE DATABASE ML_PROJECT;

-- Create schemas for different data stages
CREATE SCHEMA IF NOT EXISTS RAW COMMENT = 'Landing zone for raw source data';
CREATE SCHEMA IF NOT EXISTS STAGING COMMENT = 'Cleaned and validated data';
CREATE SCHEMA IF NOT EXISTS FEATURES COMMENT = 'Feature store for ML';
CREATE SCHEMA IF NOT EXISTS MODELS COMMENT = 'Model registry, UDFs, procedures';
CREATE SCHEMA IF NOT EXISTS ANALYTICS COMMENT = 'Dashboards, reports, metrics';

-- =============================================================================
-- WAREHOUSES
-- =============================================================================

-- ETL warehouse for data processing
CREATE WAREHOUSE IF NOT EXISTS ETL_WH
    WITH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for ETL and data processing tasks';

-- ML warehouse for model training (larger for compute)
CREATE WAREHOUSE IF NOT EXISTS ML_WH
    WITH
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND = 120
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for ML model training';

-- App warehouse for dashboards and queries
CREATE WAREHOUSE IF NOT EXISTS APP_WH
    WITH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for Streamlit apps and queries';

-- =============================================================================
-- ROLES AND PERMISSIONS
-- =============================================================================

-- Create role for data scientists (if not exists)
CREATE ROLE IF NOT EXISTS DATA_SCIENTIST;

-- Grant permissions
GRANT USAGE ON DATABASE ML_PROJECT TO ROLE DATA_SCIENTIST;
GRANT USAGE ON ALL SCHEMAS IN DATABASE ML_PROJECT TO ROLE DATA_SCIENTIST;
GRANT ALL ON SCHEMA ML_PROJECT.RAW TO ROLE DATA_SCIENTIST;
GRANT ALL ON SCHEMA ML_PROJECT.STAGING TO ROLE DATA_SCIENTIST;
GRANT ALL ON SCHEMA ML_PROJECT.FEATURES TO ROLE DATA_SCIENTIST;
GRANT ALL ON SCHEMA ML_PROJECT.MODELS TO ROLE DATA_SCIENTIST;
GRANT ALL ON SCHEMA ML_PROJECT.ANALYTICS TO ROLE DATA_SCIENTIST;

-- Grant warehouse usage
GRANT USAGE ON WAREHOUSE ETL_WH TO ROLE DATA_SCIENTIST;
GRANT USAGE ON WAREHOUSE ML_WH TO ROLE DATA_SCIENTIST;
GRANT USAGE ON WAREHOUSE APP_WH TO ROLE DATA_SCIENTIST;

-- Grant future permissions
GRANT ALL ON FUTURE TABLES IN SCHEMA ML_PROJECT.RAW TO ROLE DATA_SCIENTIST;
GRANT ALL ON FUTURE TABLES IN SCHEMA ML_PROJECT.STAGING TO ROLE DATA_SCIENTIST;
GRANT ALL ON FUTURE TABLES IN SCHEMA ML_PROJECT.FEATURES TO ROLE DATA_SCIENTIST;
GRANT ALL ON FUTURE TABLES IN SCHEMA ML_PROJECT.MODELS TO ROLE DATA_SCIENTIST;
GRANT ALL ON FUTURE TABLES IN SCHEMA ML_PROJECT.ANALYTICS TO ROLE DATA_SCIENTIST;

-- =============================================================================
-- FILE FORMATS
-- =============================================================================

USE SCHEMA RAW;

-- CSV file format for data ingestion
CREATE OR REPLACE FILE FORMAT CSV_FORMAT
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF = ('', 'NULL', 'null', 'NA', 'N/A')
    TRIM_SPACE = TRUE
    ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE;

-- JSON file format for metadata
CREATE OR REPLACE FILE FORMAT JSON_FORMAT
    TYPE = 'JSON'
    STRIP_OUTER_ARRAY = TRUE;

-- =============================================================================
-- STAGES
-- =============================================================================

-- Internal stage for file uploads
CREATE OR REPLACE STAGE ML_DATA_STAGE
    FILE_FORMAT = CSV_FORMAT
    COMMENT = 'Internal stage for ML data uploads';

-- =============================================================================
-- TABLES (Template - customize for your dataset)
-- =============================================================================

-- Model metrics tracking table
CREATE OR REPLACE TABLE ANALYTICS.MODEL_METRICS (
    MODEL_NAME VARCHAR(100),
    MODEL_VERSION VARCHAR(50),
    ACCURACY FLOAT,
    PRECISION FLOAT,
    RECALL FLOAT,
    F1 FLOAT,
    AUC_ROC FLOAT,
    TRAINING_ROWS INTEGER,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Prediction log table
CREATE OR REPLACE TABLE ANALYTICS.PREDICTION_LOG (
    PREDICTION_ID VARCHAR(36) DEFAULT UUID_STRING(),
    MODEL_NAME VARCHAR(100),
    MODEL_VERSION VARCHAR(50),
    INPUT_FEATURES VARIANT,
    PREDICTION INTEGER,
    PROBABILITY FLOAT,
    LATENCY_MS FLOAT,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Drift monitoring table
CREATE OR REPLACE TABLE ANALYTICS.DRIFT_ALERTS (
    ALERT_ID VARCHAR(36) DEFAULT UUID_STRING(),
    FEATURE_NAME VARCHAR(100),
    DRIFT_SCORE FLOAT,
    P_VALUE FLOAT,
    BASELINE_MEAN FLOAT,
    CURRENT_MEAN FLOAT,
    ALERT_TIME TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- =============================================================================
-- VERIFICATION
-- =============================================================================

-- Verify setup
SHOW DATABASES LIKE 'ML_PROJECT';
SHOW SCHEMAS IN DATABASE ML_PROJECT;
SHOW WAREHOUSES LIKE '%_WH';
SHOW TABLES IN DATABASE ML_PROJECT;
