-- Snowflake UDFs for ML Model Inference
-- Template file - customize for your model

USE DATABASE ML_PROJECT;
USE SCHEMA MODELS;
USE WAREHOUSE ML_WH;

-- =============================================================================
-- EXAMPLE: SCALAR UDF Template
-- =============================================================================

-- Template prediction UDF - customize for your features
CREATE OR REPLACE FUNCTION PREDICT_TEMPLATE(
    FEATURE_1 FLOAT,
    FEATURE_2 FLOAT,
    FEATURE_3 INTEGER
)
RETURNS INTEGER
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python', 'pandas')
HANDLER = 'predict'
AS
$$
def predict(feature_1, feature_2, feature_3):
    import pandas as pd
    from snowflake.snowpark import Session
    from snowflake.ml.registry import Registry

    # Create input DataFrame
    data = pd.DataFrame([{
        'FEATURE_1': feature_1,
        'FEATURE_2': feature_2,
        'FEATURE_3': feature_3
    }])

    # Get session and load model
    session = Session.builder.getOrCreate()
    registry = Registry(session=session)

    try:
        model = registry.get_model('YOUR_MODEL_NAME').default
        prediction = model.run(session.create_dataframe(data), function_name='predict')
        return int(prediction.collect()[0]['PREDICTION'])
    except Exception as e:
        return -1  # Error indicator
$$;

-- =============================================================================
-- BATCH PREDICTION PROCEDURE TEMPLATE
-- =============================================================================

CREATE OR REPLACE PROCEDURE BATCH_PREDICT(
    SOURCE_TABLE VARCHAR,
    TARGET_TABLE VARCHAR
)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
HANDLER = 'batch_predict'
AS
$$
def batch_predict(session, source_table, target_table):
    from snowflake.ml.registry import Registry

    # Load data
    df = session.table(source_table)
    row_count = df.count()

    # Load model
    registry = Registry(session=session)
    model = registry.get_model('YOUR_MODEL_NAME').default

    # Predict
    predictions = model.run(df, function_name='predict')

    # Save results
    predictions.write.mode('overwrite').save_as_table(target_table)

    return f"Successfully predicted {row_count} rows. Results saved to {target_table}"
$$;

-- =============================================================================
-- USAGE EXAMPLES
-- =============================================================================

-- Example: Single prediction
-- SELECT MODELS.PREDICT_TEMPLATE(1.5, 2.3, 1);

-- Example: Batch predictions on a table
-- CALL MODELS.BATCH_PREDICT('STAGING.INPUT_DATA', 'ANALYTICS.PREDICTIONS');
