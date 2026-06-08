"""
load.py — Warehouse load functions for the e-commerce ETL pipeline.

Handles three target tables:
  - fact_order_items       (UPSERT on order_item_id)
  - daily_category_sales   (DELETE-then-INSERT on summary_date)
  - etl_run_log            (INSERT one audit row per run)

Also owns the read_parquet() and validate() steps so the load phase is
fully decoupled from the in-memory transform DataFrame.

Import and call from etl_pipeline.py or any orchestrator.
"""

from __future__ import annotations

from typing import Optional

import psycopg2
import psycopg2.extras
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


# -------------------------
# CONNECTION HELPER
# -------------------------

def _get_conn(dw: dict) -> psycopg2.extensions.connection:
    """
    Open and return a psycopg2 connection to the warehouse.

    Args:
        dw: Dict with keys host, port, db, user, password.

    Returns:
        Open psycopg2 connection (autocommit=False).
    """
    return psycopg2.connect(
        host=dw["host"],
        port=dw["port"],
        dbname=dw["db"],
        user=dw["user"],
        password=dw["password"],
    )


# -------------------------
# READ PARQUET
# -------------------------

def read_parquet(spark: SparkSession, fact_path: str, summary_path: str) -> tuple[DataFrame, DataFrame]:
    """
    Read the fact and summary Parquet files written by write_parquet().

    Decouples the load step from the in-memory transform DataFrame so each
    step can be re-run independently without re-running the full pipeline.

    Args:
        spark:        Active SparkSession.
        fact_path:    Path to the fact Parquet directory (e.g. data/processed/dt=.../fact).
        summary_path: Path to the summary Parquet directory.

    Returns:
        Tuple of (fact DataFrame, summary DataFrame).
    """
    fact    = spark.read.parquet(fact_path)
    summary = spark.read.parquet(summary_path)

    print(f"[read_parquet] Fact rows:    {fact.count()}")
    print(f"[read_parquet] Summary rows: {summary.count()}")

    return fact, summary


# -------------------------
# VALIDATE
# -------------------------

def validate(fact: DataFrame, partition_date: str) -> None:
    """
    Run post-load data quality assertions against the loaded fact DataFrame.

    Assertions (all must pass; raises ValueError on first failure):
      1. COUNT(*) > 0 for the loaded partition date.
      2. No rows with gross_revenue < 0.
      3. All non-null margin_pct values fall within [-100, 100].

    Args:
        fact:           Fact DataFrame read back from Parquet after loading.
        partition_date: 'YYYY-MM-DD' string — only rows for this date are checked.

    Raises:
        ValueError: If any assertion fails, with a descriptive message.
    """
    date_rows = fact.filter(F.col("order_date") == partition_date)

    # 1. Row count > 0
    count = date_rows.count()
    if count == 0:
        raise ValueError(
            f"[validate] FAILED: COUNT(*) = 0 for partition {partition_date}. "
            "No rows were loaded."
        )
    print(f"[validate] COUNT(*) = {count} for {partition_date}. OK.")

    # 2. No negative gross_revenue
    neg_rev = date_rows.filter(F.col("gross_revenue") < 0).count()
    if neg_rev > 0:
        raise ValueError(
            f"[validate] FAILED: {neg_rev} rows have negative gross_revenue "
            f"for partition {partition_date}."
        )
    print(f"[validate] No negative gross_revenue. OK.")

    # 3. margin_pct in [-100, 100] (nulls are allowed and skipped)
    bad_margin = (
        date_rows
        .filter(F.col("margin_pct").isNotNull())
        .filter((F.col("margin_pct") < -100) | (F.col("margin_pct") > 100))
        .count()
    )
    if bad_margin > 0:
        raise ValueError(
            f"[validate] FAILED: {bad_margin} rows have margin_pct outside "
            f"[-100, 100] for partition {partition_date}."
        )
    print(f"[validate] All margin_pct values in [-100, 100]. OK.")


# -------------------------
# LOAD: fact_order_items
# -------------------------

def load_fact_order_items(
    fact: DataFrame,
    jdbc_url: str,
    jdbc_props: dict,
    dw: dict,
) -> int:
    """
    Upsert fact_order_items rows into the warehouse.

    Strategy: write the batch to a temp table via JDBC, then run a single
    INSERT ... ON CONFLICT (order_item_id) DO UPDATE in psycopg2 to merge
    into the real table. This handles re-runs of the same date without
    creating duplicate rows.

    The loaded_at column is set to NOW() by the database default on insert
    and is NOT updated on conflict (existing timestamps are preserved).

    Args:
        fact:       Transformed DataFrame read from Parquet.
        jdbc_url:   JDBC connection string for the warehouse DB.
        jdbc_props: Dict with 'user', 'password', 'driver'.
        dw:         Dict with host, port, db, user, password for psycopg2.

    Returns:
        Number of rows upserted.
    """
    staging_table = "fact_order_items_staging"

    (
        fact.write
        .jdbc(
            url=jdbc_url,
            table=staging_table,
            mode="overwrite",
            properties=jdbc_props,
        )
    )

    upsert_sql = f"""
        INSERT INTO fact_order_items (
            order_item_id, order_id, order_date,
            customer_id, customer_tier, customer_country,
            product_id, product_category,
            quantity, unit_price, unit_cost,
            gross_revenue, gross_margin, margin_pct,
            channel, etl_run_id
        )
        SELECT
            order_item_id::BIGINT, order_id::BIGINT, order_date::DATE,
            customer_id::BIGINT, customer_tier, customer_country,
            product_id::BIGINT, product_category,
            quantity::INTEGER, unit_price::NUMERIC, unit_cost::NUMERIC,
            gross_revenue::NUMERIC, gross_margin::NUMERIC, margin_pct::NUMERIC,
            channel, etl_run_id
        FROM {staging_table}
        ON CONFLICT (order_item_id) DO UPDATE SET
            order_id         = EXCLUDED.order_id,
            order_date       = EXCLUDED.order_date,
            customer_id      = EXCLUDED.customer_id,
            customer_tier    = EXCLUDED.customer_tier,
            customer_country = EXCLUDED.customer_country,
            product_id       = EXCLUDED.product_id,
            product_category = EXCLUDED.product_category,
            quantity         = EXCLUDED.quantity,
            unit_price       = EXCLUDED.unit_price,
            unit_cost        = EXCLUDED.unit_cost,
            gross_revenue    = EXCLUDED.gross_revenue,
            gross_margin     = EXCLUDED.gross_margin,
            margin_pct       = EXCLUDED.margin_pct,
            channel          = EXCLUDED.channel,
            etl_run_id       = EXCLUDED.etl_run_id;
    """

    conn = _get_conn(dw)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(upsert_sql)
                row_count = cur.rowcount
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {staging_table}")
    finally:
        conn.close()

    print(f"[fact_order_items] Upserted {row_count} rows.")
    return row_count


# -------------------------
# LOAD: daily_category_sales
# -------------------------

def load_daily_category_sales(
    summary: DataFrame,
    jdbc_url: str,
    jdbc_props: dict,
    dw: dict,
    partition_date: str,
    etl_run_id: str,
) -> int:
    """
    DELETE-then-INSERT daily_category_sales for the given partition date.

    Idempotent: deletes all rows for partition_date before inserting the
    freshly aggregated summary read from Parquet.

    Args:
        summary:        Summary DataFrame read from Parquet (summary_path).
        jdbc_url:       JDBC connection string for the warehouse DB.
        jdbc_props:     Dict with 'user', 'password', 'driver'.
        dw:             Dict with host, port, db, user, password for psycopg2.
        partition_date: 'YYYY-MM-DD' date being processed.
        etl_run_id:     UUID string to tag the inserted rows.

    Returns:
        Number of rows inserted.
    """
    summary = summary.withColumn("etl_run_id", F.lit(etl_run_id))
    row_count = summary.count()

    conn = _get_conn(dw)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM daily_category_sales WHERE summary_date = %s",
                    (partition_date,),
                )
                deleted = cur.rowcount
    except psycopg2.errors.UndefinedTable:
        deleted = 0
        print("[daily_category_sales] Table does not exist yet — skipping delete.")
    finally:
        conn.close()

    if deleted:
        print(f"[daily_category_sales] Deleted {deleted} stale rows for {partition_date}.")

    (
        summary.write
        .jdbc(
            url=jdbc_url,
            table="daily_category_sales",
            mode="append",
            properties=jdbc_props,
        )
    )

    print(f"[daily_category_sales] Inserted {row_count} rows for {partition_date}.")
    return row_count


# -------------------------
# LOAD: etl_run_log
# -------------------------

def log_run_start(
    dw: dict,
    run_id: str,
    dag_id: str,
    execution_date: str,
) -> None:
    """
    Insert a RUNNING row into etl_run_log at the start of the pipeline.

    Args:
        dw:             Dict with host, port, db, user, password.
        run_id:         UUID string matching the etl_run_id on fact rows.
        dag_id:         Airflow DAG identifier (or 'manual' for local runs).
        execution_date: Logical date being processed ('YYYY-MM-DD').
    """
    sql = """
        INSERT INTO etl_run_log
            (run_id, dag_id, execution_date, started_at, status)
        VALUES
            (%s, %s, %s, NOW(), 'RUNNING')
        ON CONFLICT (run_id) DO NOTHING;
    """
    conn = _get_conn(dw)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (run_id, dag_id, execution_date))
    finally:
        conn.close()

    print(f"[etl_run_log] Run {run_id} started.")


def log_run_end(
    dw: dict,
    run_id: str,
    status: str,
    rows_extracted: int,
    rows_loaded: int,
    error_message: Optional[str] = None,
) -> None:
    """
    Update etl_run_log with the final status when the pipeline finishes.

    Args:
        dw:             Dict with host, port, db, user, password.
        run_id:         UUID string matching the row inserted by log_run_start().
        status:         'SUCCESS' or 'FAILED'.
        rows_extracted: Total rows read from the fact Parquet file.
        rows_loaded:    Fact rows written to the warehouse.
        error_message:  Exception message on failure; None on success.
    """
    sql = """
        UPDATE etl_run_log
        SET
            finished_at    = NOW(),
            status         = %s,
            rows_extracted = %s,
            rows_loaded    = %s,
            error_message  = %s
        WHERE run_id = %s;
    """
    conn = _get_conn(dw)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (status, rows_extracted, rows_loaded, error_message, run_id))
    finally:
        conn.close()

    print(f"[etl_run_log] Run {run_id} → {status} "
          f"(extracted={rows_extracted}, loaded={rows_loaded}).")


# -------------------------
# ORCHESTRATED LOAD
# -------------------------

def run_all_loads(
    spark: SparkSession,
    fact_path: str,
    summary_path: str,
    jdbc_url: str,
    jdbc_props: dict,
    dw: dict,
    partition_date: str,
    etl_run_id: str,
    dag_id: str = "manual",
) -> dict:
    """
    Read Parquet outputs, run all three load steps, validate, and audit.

    Reads fact and summary DataFrames from Parquet (written by write_parquet())
    so this step is independently re-runnable without re-running transform.
    Validation runs after loading and raises on failure, which is caught here
    and logged as FAILED in etl_run_log before re-raising.

    Args:
        spark:          Active SparkSession (needed to read Parquet).
        fact_path:      Path to the fact Parquet directory.
        summary_path:   Path to the summary Parquet directory.
        jdbc_url:       JDBC connection string for the warehouse DB.
        jdbc_props:     Dict with 'user', 'password', 'driver'.
        dw:             Dict with host, port, db, user, password.
        partition_date: 'YYYY-MM-DD' date being processed.
        etl_run_id:     UUID string shared across all three tables for this run.
        dag_id:         Airflow DAG ID or 'manual' for local runs.

    Returns:
        Dict with keys 'rows_extracted' and 'rows_loaded'.
    """
    log_run_start(dw, run_id=etl_run_id, dag_id=dag_id, execution_date=partition_date)

    fact, summary = read_parquet(spark, fact_path, summary_path)
    rows_extracted = fact.count()
    rows_loaded = 0

    try:
        rows_loaded = load_fact_order_items(fact, jdbc_url, jdbc_props, dw)
        load_daily_category_sales(summary, jdbc_url, jdbc_props, dw, partition_date, etl_run_id)
        validate(fact, partition_date)
        log_run_end(
            dw,
            run_id=etl_run_id,
            status="SUCCESS",
            rows_extracted=rows_extracted,
            rows_loaded=rows_loaded,
        )
    except Exception as exc:
        log_run_end(
            dw,
            run_id=etl_run_id,
            status="FAILED",
            rows_extracted=rows_extracted,
            rows_loaded=rows_loaded,
            error_message=str(exc),
        )
        raise

    return {"rows_extracted": rows_extracted, "rows_loaded": rows_loaded}