"""
Unit tests for load.py

Scope:
  - read_parquet, validate: fully testable with a local SparkSession.
  - load_fact_order_items, load_daily_category_sales: JDBC writes and
    psycopg2 connections are mocked; SQL strings and call signatures are
    asserted where they encode real business logic.
  - log_run_start, log_run_end: psycopg2 is mocked; SQL content and
    parameter ordering are verified.
  - run_all_loads: all I/O mocked; orchestration logic (order of calls,
    error path, return value) is under test.

Run:
    pytest test_load.py -v
"""

from __future__ import annotations

import os
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# ---------------------------------------------------------------------------
# Shared SparkSession
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    os.environ["PYSPARK_PYTHON"] = r"C:\Users\DELL\AppData\Local\Programs\Python\Python311\python.exe"
    os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\DELL\AppData\Local\Programs\Python\Python311\python.exe"
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    session = (
        SparkSession.builder
        .appName("load_unit_tests")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# Schema + DataFrame factory helpers
# ---------------------------------------------------------------------------

FACT_SCHEMA = StructType([
    StructField("order_item_id",    LongType(),    True),
    StructField("order_id",         LongType(),    True),
    StructField("order_date",       DateType(),    True),
    StructField("customer_id",      LongType(),    True),
    StructField("customer_tier",    StringType(),  True),
    StructField("customer_country", StringType(),  True),
    StructField("product_id",       LongType(),    True),
    StructField("product_category", StringType(),  True),
    StructField("quantity",         IntegerType(), True),
    StructField("unit_price",       DoubleType(),  True),
    StructField("unit_cost",        DoubleType(),  True),
    StructField("gross_revenue",    DoubleType(),  True),
    StructField("gross_margin",     DoubleType(),  True),
    StructField("margin_pct",       DoubleType(),  True),
    StructField("channel",          StringType(),  True),
    StructField("etl_run_id",       StringType(),  True),
])

SUMMARY_SCHEMA = StructType([
    StructField("summary_date",      DateType(),    True),
    StructField("product_category",  StringType(),  True),
    StructField("customer_country",  StringType(),  True),
    StructField("order_count",       LongType(),    True),
    StructField("units_sold",        IntegerType(), True),
    StructField("gross_revenue",     DoubleType(),  True),
    StructField("gross_margin",      DoubleType(),  True),
    StructField("avg_order_value",   DoubleType(),  True),
])


def make_fact(spark, rows=None):
    if rows is None:
        rows = [
            (1001, 1, date(2026, 6, 1), 1, "gold", "US",
             10, "electronics", 2, 100.0, 50.0, 200.0, 100.0, 50.0, "web", "run-1"),
            (1002, 2, date(2026, 6, 1), 2, "silver", "UK",
             20, "apparel",     1,  50.0, 15.0,  50.0,  35.0, 70.0, "app", "run-1"),
        ]
    return spark.createDataFrame(rows, schema=FACT_SCHEMA)


def make_summary(spark, rows=None):
    if rows is None:
        rows = [
            (date(2026, 6, 1), "electronics", "US", 1, 2, 200.0, 100.0, 200.0),
            (date(2026, 6, 1), "apparel",     "UK", 1, 1,  50.0,  35.0,  50.0),
        ]
    return spark.createDataFrame(rows, schema=SUMMARY_SCHEMA)


DW = {
    "host": "localhost",
    "port": "5432",
    "db":   "ecommerce_dw",
    "user": "postgres",
    "password": "admin",
}
JDBC_URL   = "jdbc:postgresql://localhost:5432/ecommerce_dw"
JDBC_PROPS = {"user": "postgres", "password": "admin", "driver": "org.postgresql.Driver"}


# ---------------------------------------------------------------------------
# read_parquet
# ---------------------------------------------------------------------------

class TestReadParquet:

    def test_returns_two_dataframes(self, spark, tmp_path):
        from load import read_parquet

        fact_path    = str(tmp_path / "fact")
        summary_path = str(tmp_path / "summary")
        make_fact(spark).write.mode("overwrite").parquet(fact_path)
        make_summary(spark).write.mode("overwrite").parquet(summary_path)

        fact, summary = read_parquet(spark, fact_path, summary_path)
        assert fact.count() == 2
        assert summary.count() == 2

    def test_fact_columns_preserved(self, spark, tmp_path):
        from load import read_parquet

        fact_path    = str(tmp_path / "fact")
        summary_path = str(tmp_path / "summary")
        make_fact(spark).write.mode("overwrite").parquet(fact_path)
        make_summary(spark).write.mode("overwrite").parquet(summary_path)

        fact, _ = read_parquet(spark, fact_path, summary_path)
        assert "gross_revenue" in fact.columns
        assert "etl_run_id"    in fact.columns

    def test_missing_path_raises(self, spark, tmp_path):
        from load import read_parquet

        with pytest.raises(Exception):
            read_parquet(spark, str(tmp_path / "nonexistent_fact"), str(tmp_path / "nonexistent_summary"))


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class TestValidate:

    PARTITION = "2026-06-01"

    def test_valid_fact_passes(self, spark):
        from load import validate
        validate(make_fact(spark), self.PARTITION)   # should not raise

    def test_empty_partition_raises(self, spark):
        from load import validate
        # Use a date that has no rows
        with pytest.raises(ValueError, match="COUNT\\(\\*\\) = 0"):
            validate(make_fact(spark), "2025-01-01")

    def test_negative_gross_revenue_raises(self, spark):
        from load import validate
        rows = [
            (1001, 1, date(2026, 6, 1), 1, "gold", "US",
             10, "electronics", 2, 100.0, 50.0, -1.0, 100.0, 50.0, "web", "run-1"),
        ]
        fact = make_fact(spark, rows)
        with pytest.raises(ValueError, match="negative gross_revenue"):
            validate(fact, self.PARTITION)

    def test_zero_gross_revenue_is_allowed(self, spark):
        from load import validate
        rows = [
            (1001, 1, date(2026, 6, 1), 1, "gold", "US",
             10, "electronics", 2, 100.0, 50.0, 0.0, 0.0, 0.0, "web", "run-1"),
        ]
        validate(make_fact(spark, rows), self.PARTITION)   # should not raise

    def test_margin_pct_above_100_raises(self, spark):
        from load import validate
        rows = [
            (1001, 1, date(2026, 6, 1), 1, "gold", "US",
             10, "electronics", 2, 100.0, 50.0, 200.0, 100.0, 101.0, "web", "run-1"),
        ]
        with pytest.raises(ValueError, match="margin_pct outside"):
            validate(make_fact(spark, rows), self.PARTITION)

    def test_margin_pct_below_minus_100_raises(self, spark):
        from load import validate
        rows = [
            (1001, 1, date(2026, 6, 1), 1, "gold", "US",
             10, "electronics", 2, 100.0, 50.0, 200.0, 100.0, -101.0, "web", "run-1"),
        ]
        with pytest.raises(ValueError, match="margin_pct outside"):
            validate(make_fact(spark, rows), self.PARTITION)

    def test_null_margin_pct_is_allowed(self, spark):
        """Nulls in margin_pct must be skipped, not treated as violations."""
        from load import validate
        rows = [
            (1001, 1, date(2026, 6, 1), 1, "gold", "US",
             10, "electronics", 2, 100.0, 50.0, 200.0, 100.0, None, "web", "run-1"),
        ]
        validate(make_fact(spark, rows), self.PARTITION)   # should not raise

    def test_margin_pct_at_boundary_100_is_valid(self, spark):
        from load import validate
        rows = [
            (1001, 1, date(2026, 6, 1), 1, "gold", "US",
             10, "electronics", 2, 100.0, 50.0, 200.0, 100.0, 100.0, "web", "run-1"),
        ]
        validate(make_fact(spark, rows), self.PARTITION)   # exactly 100 — boundary inclusive

    def test_margin_pct_at_boundary_minus_100_is_valid(self, spark):
        from load import validate
        rows = [
            (1001, 1, date(2026, 6, 1), 1, "gold", "US",
             10, "electronics", 2, 100.0, 50.0, 200.0, 100.0, -100.0, "web", "run-1"),
        ]
        validate(make_fact(spark, rows), self.PARTITION)

    def test_only_checks_rows_for_given_partition_date(self, spark):
        """Rows from a different date with bad margin_pct must not trigger failure."""
        from load import validate
        rows = [
            # Good row for the partition under test
            (1001, 1, date(2026, 6, 1), 1, "gold", "US",
             10, "electronics", 2, 100.0, 50.0, 200.0, 100.0, 50.0, "web", "run-1"),
            # Bad margin_pct on a DIFFERENT date — should be ignored
            (1002, 2, date(2026, 5, 1), 2, "silver", "UK",
             20, "apparel",     1,  50.0, 15.0,  50.0,  35.0, 999.0, "app", "run-1"),
        ]
        fact = make_fact(spark, rows)
        validate(fact, "2026-06-01")   # should not raise


# ---------------------------------------------------------------------------
# load_fact_order_items
# ---------------------------------------------------------------------------

class TestLoadFactOrderItems:
    """
    JDBC writes and psycopg2 are mocked. Tests verify:
      - staging write is called with 'overwrite' mode
      - the upsert SQL targets the correct tables
      - the staging table is dropped after upsert
      - the row count returned is whatever psycopg2 reports
    """

    def _run(self, spark, mock_conn, rowcount=2):
        from load import load_fact_order_items

        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.rowcount = rowcount

        mock_conn_inst = MagicMock()
        mock_conn_inst.__enter__ = lambda s: s
        mock_conn_inst.__exit__ = MagicMock(return_value=False)
        mock_conn_inst.cursor.return_value = mock_cur
        mock_conn.return_value = mock_conn_inst

        fact = make_fact(spark)

        with patch.object(fact, "write") as mock_write:
            mock_write.jdbc = MagicMock()
            result = load_fact_order_items(fact, JDBC_URL, JDBC_PROPS, DW)

        return result, mock_write, mock_cur

    @patch("load._get_conn")
    def test_returns_rowcount_from_cursor(self, mock_conn, spark):
        result, _, _ = self._run(spark, mock_conn, rowcount=5)
        assert result == 5

    @patch("load._get_conn")
    def test_jdbc_write_uses_overwrite_mode(self, mock_conn, spark):
        _, mock_write, _ = self._run(spark, mock_conn)
        call_kwargs = mock_write.jdbc.call_args
        assert call_kwargs.kwargs.get("mode") == "overwrite" or "overwrite" in call_kwargs.args

    @patch("load._get_conn")
    def test_upsert_sql_targets_fact_table(self, mock_conn, spark):
        _, _, mock_cur = self._run(spark, mock_conn)
        executed_sqls = [str(c.args[0]) for c in mock_cur.execute.call_args_list]
        assert any("fact_order_items" in sql for sql in executed_sqls)

    @patch("load._get_conn")
    def test_upsert_sql_contains_on_conflict(self, mock_conn, spark):
        _, _, mock_cur = self._run(spark, mock_conn)
        executed_sqls = [str(c.args[0]).upper() for c in mock_cur.execute.call_args_list]
        assert any("ON CONFLICT" in sql for sql in executed_sqls)

    @patch("load._get_conn")
    def test_staging_table_dropped_after_upsert(self, mock_conn, spark):
        _, _, mock_cur = self._run(spark, mock_conn)
        executed_sqls = [str(c.args[0]).upper() for c in mock_cur.execute.call_args_list]
        assert any("DROP TABLE" in sql and "STAGING" in sql for sql in executed_sqls)

    @patch("load._get_conn")
    def test_connection_closed_on_success(self, mock_conn, spark):
        _, _, _ = self._run(spark, mock_conn)
        mock_conn.return_value.close.assert_called()


# ---------------------------------------------------------------------------
# load_daily_category_sales
# ---------------------------------------------------------------------------

class TestLoadDailyCategorySales:

    PARTITION = "2026-06-01"
    RUN_ID    = "run-test"

    def _run(self, spark, mock_conn):
        from load import load_daily_category_sales

        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.rowcount = 2

        mock_conn_inst = MagicMock()
        mock_conn_inst.__enter__ = lambda s: s
        mock_conn_inst.__exit__ = MagicMock(return_value=False)
        mock_conn_inst.cursor.return_value = mock_cur
        mock_conn.return_value = mock_conn_inst

        summary = make_summary(spark)

        with patch.object(summary, "write") as mock_write:
            mock_write.jdbc = MagicMock()
            result = load_daily_category_sales(
                summary, JDBC_URL, JDBC_PROPS, DW, self.PARTITION, self.RUN_ID
            )

        return result, mock_write, mock_cur

    @patch("load._get_conn")
    def test_returns_row_count(self, mock_conn, spark):
        result, _, _ = self._run(spark, mock_conn)
        assert result == 2   # matches make_summary default rows

    @patch("load._get_conn")
    def test_delete_issued_before_insert(self, mock_conn, spark):
        _, _, mock_cur = self._run(spark, mock_conn)
        first_sql = str(mock_cur.execute.call_args_list[0].args[0]).upper()
        assert "DELETE" in first_sql

    @patch("load._get_conn")
    def test_delete_uses_partition_date_param(self, mock_conn, spark):
        _, _, mock_cur = self._run(spark, mock_conn)
        first_call = mock_cur.execute.call_args_list[0]
        # Second positional arg is the params tuple
        params = first_call.args[1] if len(first_call.args) > 1 else first_call.kwargs.get("vars")
        assert self.PARTITION in params

    @patch("load._get_conn")
    def test_jdbc_append_used_for_insert(self, mock_conn, spark):
        _, mock_write, _ = self._run(spark, mock_conn)
        call_kwargs = mock_write.jdbc.call_args
        assert call_kwargs.kwargs.get("mode") == "append" or "append" in call_kwargs.args

    @patch("load._get_conn")
    def test_etl_run_id_column_added_to_summary(self, mock_conn, spark):
        """summary DataFrame must have etl_run_id column added before write."""
        from load import load_daily_category_sales
        import pyspark.sql.functions as F

        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.rowcount = 2

        mock_conn_inst = MagicMock()
        mock_conn_inst.__enter__ = lambda s: s
        mock_conn_inst.__exit__ = MagicMock(return_value=False)
        mock_conn_inst.cursor.return_value = mock_cur
        mock_conn.return_value = mock_conn_inst

        summary = make_summary(spark)
        captured = {}

        def capture_jdbc(**kwargs):
            captured["df"] = kwargs.get("table")  # we'll inspect via the write mock

        # Intercept the DataFrame passed to .write.jdbc to check its columns
        written_dfs = []
        original_jdbc = None

        class WriteMock:
            def jdbc(self, *args, **kwargs):
                pass

        with patch.object(summary.__class__, "withColumn", wraps=summary.withColumn) as mock_wc:
            with patch.object(summary, "write") as mock_write:
                mock_write.jdbc = MagicMock()
                load_daily_category_sales(
                    summary, JDBC_URL, JDBC_PROPS, DW, self.PARTITION, self.RUN_ID
                )
            # withColumn should have been called with "etl_run_id"
            call_args = [c.args[0] for c in mock_wc.call_args_list]
            assert "etl_run_id" in call_args

    @patch("load._get_conn")
    def test_undefined_table_error_is_swallowed(self, mock_conn, spark):
        """If the target table doesn't exist yet, the DELETE should not crash."""
        import psycopg2.errors

        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.execute.side_effect = psycopg2.errors.UndefinedTable("no table")

        mock_conn_inst = MagicMock()
        mock_conn_inst.__enter__ = lambda s: s
        mock_conn_inst.__exit__ = MagicMock(return_value=False)
        mock_conn_inst.cursor.return_value = mock_cur
        mock_conn.return_value = mock_conn_inst

        summary = make_summary(spark)
        with patch.object(summary, "write") as mock_write:
            mock_write.jdbc = MagicMock()
            # Should not raise
            load_daily_category_sales(
                summary, JDBC_URL, JDBC_PROPS, DW, self.PARTITION, self.RUN_ID
            )


# ---------------------------------------------------------------------------
# log_run_start / log_run_end
# ---------------------------------------------------------------------------

class TestLogRunStart:

    @patch("load._get_conn")
    def test_inserts_running_status(self, mock_conn):
        from load import log_run_start

        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)

        mock_conn_inst = MagicMock()
        mock_conn_inst.__enter__ = lambda s: s
        mock_conn_inst.__exit__ = MagicMock(return_value=False)
        mock_conn_inst.cursor.return_value = mock_cur
        mock_conn.return_value = mock_conn_inst

        log_run_start(DW, run_id="r1", dag_id="manual", execution_date="2026-06-01")

        sql, params = mock_cur.execute.call_args.args
        assert "RUNNING" in sql
        assert "r1" in params
        assert "manual" in params
        assert "2026-06-01" in params

    @patch("load._get_conn")
    def test_on_conflict_do_nothing_present(self, mock_conn):
        """Re-running with the same run_id must not fail (ON CONFLICT DO NOTHING)."""
        from load import log_run_start

        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn_inst = MagicMock()
        mock_conn_inst.__enter__ = lambda s: s
        mock_conn_inst.__exit__ = MagicMock(return_value=False)
        mock_conn_inst.cursor.return_value = mock_cur
        mock_conn.return_value = mock_conn_inst

        log_run_start(DW, run_id="r1", dag_id="manual", execution_date="2026-06-01")

        sql = mock_cur.execute.call_args.args[0].upper()
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql

    @patch("load._get_conn")
    def test_connection_closed(self, mock_conn):
        from load import log_run_start

        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn_inst = MagicMock()
        mock_conn_inst.__enter__ = lambda s: s
        mock_conn_inst.__exit__ = MagicMock(return_value=False)
        mock_conn_inst.cursor.return_value = mock_cur
        mock_conn.return_value = mock_conn_inst

        log_run_start(DW, run_id="r1", dag_id="manual", execution_date="2026-06-01")
        mock_conn_inst.close.assert_called_once()


class TestLogRunEnd:

    def _setup_conn(self, mock_conn):
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn_inst = MagicMock()
        mock_conn_inst.__enter__ = lambda s: s
        mock_conn_inst.__exit__ = MagicMock(return_value=False)
        mock_conn_inst.cursor.return_value = mock_cur
        mock_conn.return_value = mock_conn_inst
        return mock_cur, mock_conn_inst

    @patch("load._get_conn")
    def test_success_status_written(self, mock_conn):
        from load import log_run_end
        mock_cur, _ = self._setup_conn(mock_conn)

        log_run_end(DW, run_id="r1", status="SUCCESS",
                    rows_extracted=100, rows_loaded=100)

        params = mock_cur.execute.call_args.args[1]
        assert "SUCCESS" in params

    @patch("load._get_conn")
    def test_failed_status_written(self, mock_conn):
        from load import log_run_end
        mock_cur, _ = self._setup_conn(mock_conn)

        log_run_end(DW, run_id="r1", status="FAILED",
                    rows_extracted=50, rows_loaded=0, error_message="boom")

        params = mock_cur.execute.call_args.args[1]
        assert "FAILED" in params
        assert "boom" in params

    @patch("load._get_conn")
    def test_row_counts_in_params(self, mock_conn):
        from load import log_run_end
        mock_cur, _ = self._setup_conn(mock_conn)

        log_run_end(DW, run_id="r1", status="SUCCESS",
                    rows_extracted=42, rows_loaded=41)

        params = mock_cur.execute.call_args.args[1]
        assert 42 in params
        assert 41 in params

    @patch("load._get_conn")
    def test_run_id_in_where_clause_params(self, mock_conn):
        """run_id must appear in the params (used in WHERE clause)."""
        from load import log_run_end
        mock_cur, _ = self._setup_conn(mock_conn)

        log_run_end(DW, run_id="r-xyz", status="SUCCESS",
                    rows_extracted=1, rows_loaded=1)

        params = mock_cur.execute.call_args.args[1]
        assert "r-xyz" in params

    @patch("load._get_conn")
    def test_none_error_message_on_success(self, mock_conn):
        from load import log_run_end
        mock_cur, _ = self._setup_conn(mock_conn)

        log_run_end(DW, run_id="r1", status="SUCCESS",
                    rows_extracted=10, rows_loaded=10)

        params = mock_cur.execute.call_args.args[1]
        assert None in params

    @patch("load._get_conn")
    def test_sql_is_update_not_insert(self, mock_conn):
        from load import log_run_end
        mock_cur, _ = self._setup_conn(mock_conn)

        log_run_end(DW, run_id="r1", status="SUCCESS",
                    rows_extracted=1, rows_loaded=1)

        sql = mock_cur.execute.call_args.args[0].upper()
        assert sql.strip().startswith("UPDATE")

    @patch("load._get_conn")
    def test_connection_closed(self, mock_conn):
        from load import log_run_end
        _, mock_conn_inst = self._setup_conn(mock_conn)

        log_run_end(DW, run_id="r1", status="SUCCESS",
                    rows_extracted=1, rows_loaded=1)

        mock_conn_inst.close.assert_called_once()


# ---------------------------------------------------------------------------
# run_all_loads — orchestration
# ---------------------------------------------------------------------------

class TestRunAllLoads:
    """
    All I/O is mocked. Tests verify call order, error propagation, and
    the shape of the return dict.
    """

    PARTITION = "2026-06-01"
    RUN_ID    = "run-orch"

    @pytest.fixture()
    def parquet_paths(self, spark, tmp_path):
        fact_path    = str(tmp_path / "fact")
        summary_path = str(tmp_path / "summary")
        make_fact(spark).write.mode("overwrite").parquet(fact_path)
        make_summary(spark).write.mode("overwrite").parquet(summary_path)
        return fact_path, summary_path

    def _base_patches(self):
        """Context managers that mock every external call in run_all_loads."""
        return [
            patch("load.load_fact_order_items",    return_value=2),
            patch("load.load_daily_category_sales", return_value=2),
            patch("load.log_run_start"),
            patch("load.log_run_end"),
        ]

    def test_returns_rows_extracted_and_loaded(self, spark, parquet_paths):
        from load import run_all_loads
        fact_path, summary_path = parquet_paths

        with (
            patch("load.load_fact_order_items",     return_value=2),
            patch("load.load_daily_category_sales", return_value=2),
            patch("load.log_run_start"),
            patch("load.log_run_end"),
        ):
            result = run_all_loads(
                spark=spark,
                fact_path=fact_path,
                summary_path=summary_path,
                jdbc_url=JDBC_URL,
                jdbc_props=JDBC_PROPS,
                dw=DW,
                partition_date=self.PARTITION,
                etl_run_id=self.RUN_ID,
            )

        assert result["rows_extracted"] == 2
        assert result["rows_loaded"]    == 2

    def test_log_run_start_called_before_loads(self, spark, parquet_paths):
        from load import run_all_loads
        fact_path, summary_path = parquet_paths
        call_order = []

        with (
            patch("load.log_run_start",             side_effect=lambda *a, **kw: call_order.append("start")),
            patch("load.load_fact_order_items",     side_effect=lambda *a, **kw: call_order.append("fact") or 2),
            patch("load.load_daily_category_sales", side_effect=lambda *a, **kw: call_order.append("summary") or 2),
            patch("load.log_run_end"),
            patch("load.validate"),
        ):
            run_all_loads(
                spark=spark,
                fact_path=fact_path,
                summary_path=summary_path,
                jdbc_url=JDBC_URL,
                jdbc_props=JDBC_PROPS,
                dw=DW,
                partition_date=self.PARTITION,
                etl_run_id=self.RUN_ID,
            )

        assert call_order.index("start") < call_order.index("fact")
        assert call_order.index("start") < call_order.index("summary")

    def test_log_run_end_called_with_success(self, spark, parquet_paths):
        from load import run_all_loads
        fact_path, summary_path = parquet_paths
        mock_end = MagicMock()

        with (
            patch("load.load_fact_order_items",     return_value=2),
            patch("load.load_daily_category_sales", return_value=2),
            patch("load.log_run_start"),
            patch("load.log_run_end",               mock_end),
            patch("load.validate"),
        ):
            run_all_loads(
                spark=spark,
                fact_path=fact_path,
                summary_path=summary_path,
                jdbc_url=JDBC_URL,
                jdbc_props=JDBC_PROPS,
                dw=DW,
                partition_date=self.PARTITION,
                etl_run_id=self.RUN_ID,
            )

        mock_end.assert_called_once()
        assert mock_end.call_args.kwargs["status"] == "SUCCESS"

    def test_exception_in_load_logs_failed_and_reraises(self, spark, parquet_paths):
        from load import run_all_loads
        fact_path, summary_path = parquet_paths
        mock_end = MagicMock()

        with (
            patch("load.load_fact_order_items", side_effect=RuntimeError("DB is down")),
            patch("load.load_daily_category_sales"),
            patch("load.log_run_start"),
            patch("load.log_run_end", mock_end),
            patch("load.validate"),
            pytest.raises(RuntimeError, match="DB is down"),
        ):
            run_all_loads(
                spark=spark,
                fact_path=fact_path,
                summary_path=summary_path,
                jdbc_url=JDBC_URL,
                jdbc_props=JDBC_PROPS,
                dw=DW,
                partition_date=self.PARTITION,
                etl_run_id=self.RUN_ID,
            )

        mock_end.assert_called_once()
        assert mock_end.call_args.kwargs["status"] == "FAILED"
        assert "DB is down" in mock_end.call_args.kwargs["error_message"]

    def test_validate_failure_logs_failed_and_reraises(self, spark, parquet_paths):
        from load import run_all_loads
        fact_path, summary_path = parquet_paths
        mock_end = MagicMock()

        with (
            patch("load.load_fact_order_items",     return_value=2),
            patch("load.load_daily_category_sales", return_value=2),
            patch("load.log_run_start"),
            patch("load.log_run_end", mock_end),
            patch("load.validate", side_effect=ValueError("validate exploded")),
            pytest.raises(ValueError, match="validate exploded"),
        ):
            run_all_loads(
                spark=spark,
                fact_path=fact_path,
                summary_path=summary_path,
                jdbc_url=JDBC_URL,
                jdbc_props=JDBC_PROPS,
                dw=DW,
                partition_date=self.PARTITION,
                etl_run_id=self.RUN_ID,
            )

        assert mock_end.call_args.kwargs["status"] == "FAILED"

    def test_run_id_threaded_to_log_end(self, spark, parquet_paths):
        """The same etl_run_id passed in must be forwarded to log_run_end."""
        from load import run_all_loads
        fact_path, summary_path = parquet_paths
        mock_end = MagicMock()

        with (
            patch("load.load_fact_order_items",     return_value=2),
            patch("load.load_daily_category_sales", return_value=2),
            patch("load.log_run_start"),
            patch("load.log_run_end", mock_end),
            patch("load.validate"),
        ):
            run_all_loads(
                spark=spark,
                fact_path=fact_path,
                summary_path=summary_path,
                jdbc_url=JDBC_URL,
                jdbc_props=JDBC_PROPS,
                dw=DW,
                partition_date=self.PARTITION,
                etl_run_id=self.RUN_ID,
            )

        assert mock_end.call_args.kwargs["run_id"] == self.RUN_ID

    def test_validate_called_after_loads(self, spark, parquet_paths):
        from load import run_all_loads
        fact_path, summary_path = parquet_paths
        call_order = []

        with (
            patch("load.load_fact_order_items",     side_effect=lambda *a, **kw: call_order.append("fact") or 2),
            patch("load.load_daily_category_sales", side_effect=lambda *a, **kw: call_order.append("summary") or 2),
            patch("load.log_run_start"),
            patch("load.log_run_end"),
            patch("load.validate",                  side_effect=lambda *a, **kw: call_order.append("validate")),
        ):
            run_all_loads(
                spark=spark,
                fact_path=fact_path,
                summary_path=summary_path,
                jdbc_url=JDBC_URL,
                jdbc_props=JDBC_PROPS,
                dw=DW,
                partition_date=self.PARTITION,
                etl_run_id=self.RUN_ID,
            )

        assert call_order.index("validate") > call_order.index("fact")
        assert call_order.index("validate") > call_order.index("summary")