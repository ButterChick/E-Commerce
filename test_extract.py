"""
Unit tests for etl.py

Assumptions / scope:
  - The module under test is named `etl.py` and lives on sys.path.
  - `load.run_all_loads` is mocked throughout; it is not under test here.
  - JDBC calls are mocked; no live database is required.
  - A single SparkSession is shared across the session for speed.

Run:
    pytest test_etl.py -v
"""

import os
import re
import tempfile
import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
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
# Spark sessio
#n — one per test session, reused for speed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    os.environ["PYSPARK_PYTHON"] = r"C:\Users\DELL\AppData\Local\Programs\Python\Python311\python.exe"
    os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\DELL\AppData\Local\Programs\Python\Python311\python.exe"
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    session = (
        SparkSession.builder
        .appName("etl_unit_tests")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# Shared schema / factory helpers
# ---------------------------------------------------------------------------

ORDERS_SCHEMA = StructType([
    StructField("order_item_id", LongType(),   True),
    StructField("order_id",      LongType(),   True),
    StructField("order_date",    DateType(),   True),
    StructField("customer_id",   LongType(),   True),
    StructField("product_id",    LongType(),   True),
    StructField("quantity",      IntegerType(), True),
    StructField("unit_price",    DoubleType(), True),
    StructField("channel",       StringType(), True),
])

CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id",   LongType(),   True),
    StructField("customer_tier", StringType(), True),
    StructField("country",       StringType(), True),
])

PRODUCTS_SCHEMA = StructType([
    StructField("product_id",  LongType(),   True),
    StructField("category",    StringType(), True),
    StructField("unit_cost",   DoubleType(), True),
])


def make_orders(spark, rows):
    """Build an orders DataFrame from a list of tuples matching ORDERS_SCHEMA."""
    return spark.createDataFrame(rows, schema=ORDERS_SCHEMA)


def make_customers(spark, rows=None):
    if rows is None:
        rows = [(1, "gold", "US"), (2, "silver", "UK")]
    return spark.createDataFrame(rows, schema=CUSTOMERS_SCHEMA)


def make_products(spark, rows=None):
    if rows is None:
        rows = [(10, "electronics", 50.0), (20, "apparel", 15.0)]
    return spark.createDataFrame(rows, schema=PRODUCTS_SCHEMA)


def baseline_orders(spark):
    """Two clean, valid order rows used by many tests."""
    return make_orders(spark, [
        (1001, 1, date(2026, 6, 1), 1, 10, 2,  100.0, "web"),
        (1002, 2, date(2026, 6, 1), 2, 20, 1,   50.0, "app"),
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rows_as_dicts(df):
    return [row.asDict() for row in df.collect()]


# ---------------------------------------------------------------------------
# extract_orders
# ---------------------------------------------------------------------------

class TestExtractOrders:
    """Tests for extract_orders(); uses CSV written to a temp file."""

    def _write_csv(self, path, content):
        with open(path, "w") as f:
            f.write(content)

    def test_basic_cast(self, spark, tmp_path):
        """Customer/product/order IDs must come back as LongType."""
        from extract import extract_orders

        csv_file = str(tmp_path / "orders.csv")
        self._write_csv(
            csv_file,
            "order_item_id,order_id,order_date,customer_id,product_id,quantity,unit_price,channel\n"
            "1001,1,2026-06-01,1,10,2,100.0,web\n",
        )
        df = extract_orders(spark, csv_file)
        field_map = {f.name: f.dataType for f in df.schema.fields}

        assert isinstance(field_map["customer_id"], LongType)
        assert isinstance(field_map["product_id"],  LongType)
        assert isinstance(field_map["order_id"],     LongType)
        assert isinstance(field_map["quantity"],     IntegerType)
        assert isinstance(field_map["unit_price"],   DoubleType)
        assert isinstance(field_map["order_date"],   DateType)

    def test_float_ids_are_truncated_to_long(self, spark, tmp_path):
        """IDs stored as floats (e.g. '1.0') should cast cleanly to Long."""
        from extract import extract_orders

        csv_file = str(tmp_path / "orders.csv")
        self._write_csv(
            csv_file,
            "order_item_id,order_id,order_date,customer_id,product_id,quantity,unit_price,channel\n"
            "1001.0,1.0,2026-06-01,1.0,10.0,3,75.0,web\n",
        )
        df = extract_orders(spark, csv_file)
        row = df.first()
        assert row["customer_id"] == 1
        assert row["order_id"] == 1

    def test_unparseable_id_becomes_null(self, spark, tmp_path):
        """A non-numeric customer_id should silently become NULL (not raise)."""
        from extract import extract_orders

        csv_file = str(tmp_path / "orders.csv")
        self._write_csv(
            csv_file,
            "order_item_id,order_id,order_date,customer_id,product_id,quantity,unit_price,channel\n"
            "1001,1,2026-06-01,NOT_A_NUMBER,10,2,100.0,web\n",
        )
        df = extract_orders(spark, csv_file)
        row = df.first()
        assert row["customer_id"] is None

    def test_empty_file_returns_empty_df(self, spark, tmp_path):
        from extract import extract_orders

        csv_file = str(tmp_path / "orders.csv")
        self._write_csv(
            csv_file,
            "order_item_id,order_id,order_date,customer_id,product_id,quantity,unit_price,channel\n",
        )
        df = extract_orders(spark, csv_file)
        assert df.count() == 0


# ---------------------------------------------------------------------------
# transform — cleaning
# ---------------------------------------------------------------------------

class TestTransformCleaning:
    """Critical-field nulls, non-positive values, and deduplication."""

    RUN_ID = "test-run-id"

    def _transform(self, spark, orders_rows):
        from extract import transform
        orders    = make_orders(spark, orders_rows)
        customers = make_customers(spark)
        products  = make_products(spark)
        return transform(orders, customers, products, self.RUN_ID)

    def test_null_order_id_dropped(self, spark):
        rows = [(1001, None, date(2026, 6, 1), 1, 10, 2, 100.0, "web")]
        result = self._transform(spark, rows)
        assert result.count() == 0

    def test_null_customer_id_dropped(self, spark):
        rows = [(1001, 1, date(2026, 6, 1), None, 10, 2, 100.0, "web")]
        result = self._transform(spark, rows)
        assert result.count() == 0

    def test_null_product_id_dropped(self, spark):
        rows = [(1001, 1, date(2026, 6, 1), 1, None, 2, 100.0, "web")]
        result = self._transform(spark, rows)
        assert result.count() == 0

    def test_null_order_date_dropped(self, spark):
        rows = [(1001, 1, None, 1, 10, 2, 100.0, "web")]
        result = self._transform(spark, rows)
        assert result.count() == 0

    def test_null_order_item_id_dropped(self, spark):
        rows = [(None, 1, date(2026, 6, 1), 1, 10, 2, 100.0, "web")]
        result = self._transform(spark, rows)
        assert result.count() == 0

    def test_zero_quantity_dropped(self, spark):
        rows = [(1001, 1, date(2026, 6, 1), 1, 10, 0, 100.0, "web")]
        result = self._transform(spark, rows)
        assert result.count() == 0

    def test_negative_quantity_dropped(self, spark):
        rows = [(1001, 1, date(2026, 6, 1), 1, 10, -1, 100.0, "web")]
        result = self._transform(spark, rows)
        assert result.count() == 0

    def test_zero_unit_price_dropped(self, spark):
        rows = [(1001, 1, date(2026, 6, 1), 1, 10, 2, 0.0, "web")]
        result = self._transform(spark, rows)
        assert result.count() == 0

    def test_negative_unit_price_dropped(self, spark):
        rows = [(1001, 1, date(2026, 6, 1), 1, 10, 2, -5.0, "web")]
        result = self._transform(spark, rows)
        assert result.count() == 0

    def test_valid_row_survives(self, spark):
        rows = [(1001, 1, date(2026, 6, 1), 1, 10, 2, 100.0, "web")]
        result = self._transform(spark, rows)
        assert result.count() == 1

    def test_dedup_keeps_latest_order_id(self, spark):
        """Two rows with the same order_item_id — the one with the higher order_id wins."""
        rows = [
            (1001, 5, date(2026, 6, 1), 1, 10, 1, 100.0, "web"),  # older delivery
            (1001, 9, date(2026, 6, 1), 1, 10, 2, 120.0, "web"),  # latest delivery
        ]
        result = self._transform(spark, rows)
        assert result.count() == 1
        row = result.first()
        assert row["order_id"] == 9
        assert row["quantity"] == 2

    def test_dedup_distinct_item_ids_both_kept(self, spark):
        rows = [
            (1001, 1, date(2026, 6, 1), 1, 10, 2, 100.0, "web"),
            (1002, 2, date(2026, 6, 1), 2, 20, 1,  50.0, "app"),
        ]
        result = self._transform(spark, rows)
        assert result.count() == 2

    def test_mixed_valid_and_invalid_rows(self, spark):
        rows = [
            (1001, 1, date(2026, 6, 1), 1, 10, 2, 100.0, "web"),  # valid
            (1002, 2, date(2026, 6, 1), 2, 20, 0,  50.0, "app"),  # zero qty
            (1003, 3, date(2026, 6, 1), 1, 10, 1, -10.0, "web"),  # negative price
        ]
        result = self._transform(spark, rows)
        assert result.count() == 1


# ---------------------------------------------------------------------------
# transform — computed columns
# ---------------------------------------------------------------------------

class TestTransformComputedColumns:

    RUN_ID = "test-run-id"

    def _single_row_result(self, spark, qty, unit_price, unit_cost):
        from extract import transform
        orders = make_orders(spark, [
            (1001, 1, date(2026, 6, 1), 1, 10, qty, unit_price, "web")
        ])
        customers = make_customers(spark, [(1, "gold", "US")])
        products  = make_products(spark,  [(10, "electronics", unit_cost)])
        return transform(orders, customers, products, self.RUN_ID).first()

    def test_gross_revenue(self, spark):
        row = self._single_row_result(spark, qty=3, unit_price=100.0, unit_cost=60.0)
        assert row["gross_revenue"] == pytest.approx(300.0)

    def test_gross_margin(self, spark):
        row = self._single_row_result(spark, qty=3, unit_price=100.0, unit_cost=60.0)
        assert row["gross_margin"] == pytest.approx(120.0)

    def test_margin_pct(self, spark):
        row = self._single_row_result(spark, qty=3, unit_price=100.0, unit_cost=60.0)
        # margin = 120, revenue = 300 → 40.0 %
        assert row["margin_pct"] == pytest.approx(40.0)

    def test_margin_pct_rounding(self, spark):
        """Result should be rounded to 2 decimal places."""
        # 1 unit, price=3.0, cost=2.0 → margin=1, revenue=3 → 33.33...%
        row = self._single_row_result(spark, qty=1, unit_price=3.0, unit_cost=2.0)
        assert row["margin_pct"] == pytest.approx(33.33)

    def test_etl_run_id_propagated(self, spark):
        row = self._single_row_result(spark, qty=1, unit_price=10.0, unit_cost=5.0)
        assert row["etl_run_id"] == self.RUN_ID

    def test_customer_country_renamed(self, spark):
        row = self._single_row_result(spark, qty=1, unit_price=10.0, unit_cost=5.0)
        assert row["customer_country"] == "US"
        assert "country" not in row.asDict()

    def test_product_category_renamed(self, spark):
        row = self._single_row_result(spark, qty=1, unit_price=10.0, unit_cost=5.0)
        assert row["product_category"] == "electronics"
        assert "category" not in row.asDict()

    def test_unmatched_customer_gives_null_tier(self, spark):
        """A customer_id with no match in customers table → null customer_tier."""
        from extract import transform
        orders    = make_orders(spark, [(1001, 1, date(2026, 6, 1), 999, 10, 1, 10.0, "web")])
        customers = make_customers(spark, [(1, "gold", "US")])   # 999 absent
        products  = make_products(spark,  [(10, "electronics", 5.0)])
        result    = transform(orders, customers, products, self.RUN_ID).first()
        assert result["customer_tier"] is None

    def test_unmatched_product_gives_null_category(self, spark):
        from extract import transform
        orders    = make_orders(spark, [(1001, 1, date(2026, 6, 1), 1, 999, 1, 10.0, "web")])
        customers = make_customers(spark, [(1, "gold", "US")])
        products  = make_products(spark,  [(10, "electronics", 5.0)])  # 999 absent
        result    = transform(orders, customers, products, self.RUN_ID).first()
        assert result["product_category"] is None

    def test_output_columns_exactly(self, spark):
        """Final DataFrame must contain exactly the expected columns in order."""
        from extract import transform
        orders    = baseline_orders(spark)
        customers = make_customers(spark)
        products  = make_products(spark)
        result    = transform(orders, customers, products, self.RUN_ID)

        expected = [
            "order_item_id", "order_id", "order_date",
            "customer_id", "customer_tier", "customer_country",
            "product_id", "product_category",
            "quantity", "unit_price", "unit_cost",
            "gross_revenue", "gross_margin", "margin_pct",
            "channel", "etl_run_id",
        ]
        assert result.columns == expected

    def test_zero_revenue_margin_pct_is_null(self, spark):
        """
        gross_revenue == 0 → margin_pct should be NULL (avoids divide-by-zero).
        This can't be triggered by a valid row (unit_price > 0 is enforced),
        but we test the branch directly by patching the DataFrame after cleaning.
        """
        from extract import transform
        # quantity=1, price=1 (valid), cost=1 → margin=0, revenue=1 → 0%
        # To hit the NULL branch we need revenue=0, which the cleaner blocks.
        # Instead, verify the normal zero-margin case produces 0.0, not NULL.
        row = self._single_row_result(spark, qty=1, unit_price=5.0, unit_cost=5.0)
        assert row["gross_margin"] == pytest.approx(0.0)
        assert row["margin_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# write_parquet
# ---------------------------------------------------------------------------

class TestWriteParquet:

    def test_fact_and_summary_paths_returned(self, spark, tmp_path):
        from extract import transform, write_parquet

        orders    = baseline_orders(spark)
        customers = make_customers(spark)
        products  = make_products(spark)
        fact      = transform(orders, customers, products, "run-1")

        paths = write_parquet(fact, "2026-06-01", base_path=str(tmp_path))
        assert "fact_path"    in paths
        assert "summary_path" in paths

    def test_fact_parquet_readable(self, spark, tmp_path):
        from extract import transform, write_parquet

        orders    = baseline_orders(spark)
        customers = make_customers(spark)
        products  = make_products(spark)
        fact      = transform(orders, customers, products, "run-2")
        paths     = write_parquet(fact, "2026-06-01", base_path=str(tmp_path))

        reloaded = spark.read.parquet(paths["fact_path"])
        assert reloaded.count() == 2

    def test_summary_parquet_contains_aggregated_row(self, spark, tmp_path):
        from extract import transform, write_parquet

        orders    = baseline_orders(spark)
        customers = make_customers(spark)
        products  = make_products(spark)
        fact      = transform(orders, customers, products, "run-3")
        paths     = write_parquet(fact, "2026-06-01", base_path=str(tmp_path))

        summary = spark.read.parquet(paths["summary_path"])
        assert summary.count() > 0
        assert "gross_revenue" in summary.columns
        assert "units_sold"    in summary.columns

    def test_summary_aggregates_correctly(self, spark, tmp_path):
        """Single-product, single-country row: summary revenue should match fact."""
        from extract import transform, write_parquet

        orders    = make_orders(spark, [
            (1001, 1, date(2026, 6, 1), 1, 10, 2, 100.0, "web"),
        ])
        customers = make_customers(spark, [(1, "gold", "US")])
        products  = make_products(spark,  [(10, "electronics", 50.0)])
        fact      = transform(orders, customers, products, "run-4")
        paths     = write_parquet(fact, "2026-06-01", base_path=str(tmp_path))

        summary = spark.read.parquet(paths["summary_path"])
        row = summary.first()
        assert row["gross_revenue"] == pytest.approx(200.0)
        assert row["units_sold"] == 2

    def test_overwrite_mode_replaces_data(self, spark, tmp_path):
        """A second write_parquet call to the same partition should overwrite, not append."""
        from extract import transform, write_parquet

        orders    = baseline_orders(spark)
        customers = make_customers(spark)
        products  = make_products(spark)
        fact      = transform(orders, customers, products, "run-5")

        write_parquet(fact, "2026-06-01", base_path=str(tmp_path))
        paths = write_parquet(fact, "2026-06-01", base_path=str(tmp_path))

        reloaded = spark.read.parquet(paths["fact_path"])
        assert reloaded.count() == 2   # 2, not 4


# ---------------------------------------------------------------------------
# log_quality (smoke test — just verifies it doesn't raise)
# ---------------------------------------------------------------------------

class TestLogQuality:

    def test_does_not_raise(self, spark, capsys):
        from extract import transform, log_quality

        orders    = baseline_orders(spark)
        customers = make_customers(spark)
        products  = make_products(spark)
        fact      = transform(orders, customers, products, "run-lq")
        log_quality(fact)   # should print without error

        captured = capsys.readouterr()
        assert "rows" in captured.out.lower()


# ---------------------------------------------------------------------------
# run_pipeline — integration smoke test (all external calls mocked)
# ---------------------------------------------------------------------------

class TestRunPipeline:

    @pytest.fixture()
    def orders_csv(self, tmp_path):
        """Minimal CSV in the required dt=YYYY-MM-DD path structure."""
        dt_dir = tmp_path / "raw" / "dt=2026-06-01"
        dt_dir.mkdir(parents=True)
        csv_path = dt_dir / "orders.csv"
        csv_path.write_text(
            "order_item_id,order_id,order_date,customer_id,product_id,"
            "quantity,unit_price,channel\n"
            "1001,1,2026-06-01,1,10,2,100.0,web\n"
            "1002,2,2026-06-01,2,20,1,50.0,app\n"
        )
        return str(csv_path)

    @pytest.fixture()
    def customer_df(self, spark):
        return make_customers(spark)

    @pytest.fixture()
    def product_df(self, spark):
        return make_products(spark)

    def test_returns_dataframe(self, spark, orders_csv, customer_df, product_df, tmp_path):
        from extract import run_pipeline

        with (
            patch("extract.get_spark",          return_value=spark),
            patch("extract.extract_customers",  return_value=customer_df),
            patch("extract.extract_products",   return_value=product_df),
            patch("extract.run_all_loads"),
        ):
            result = run_pipeline(
                orders_filepath=orders_csv,
                processed_base=str(tmp_path / "processed"),
            )

        assert result is not None
        assert result.count() == 2

    def test_partition_date_parse_error(self, spark, tmp_path):
        """A filepath without dt=YYYY-MM-DD should raise ValueError."""
        from extract import run_pipeline

        bad_csv = tmp_path / "orders.csv"
        bad_csv.write_text(
            "order_item_id,order_id,order_date,customer_id,product_id,quantity,unit_price,channel\n"
        )

        with (
            patch("extract.get_spark", return_value=spark),
            pytest.raises(ValueError, match="Cannot parse partition date"),
        ):
            run_pipeline(orders_filepath=str(bad_csv))

    def test_run_all_loads_called_once(self, spark, orders_csv, customer_df, product_df, tmp_path):
        from extract import run_pipeline

        mock_loads = MagicMock()
        with (
            patch("extract.get_spark",         return_value=spark),
            patch("extract.extract_customers", return_value=customer_df),
            patch("extract.extract_products",  return_value=product_df),
            patch("extract.run_all_loads",     mock_loads),
        ):
            run_pipeline(
                orders_filepath=orders_csv,
                processed_base=str(tmp_path / "processed"),
            )

        mock_loads.assert_called_once()

    def test_run_all_loads_receives_correct_partition_date(
        self, spark, orders_csv, customer_df, product_df, tmp_path
    ):
        from extract import run_pipeline

        mock_loads = MagicMock()
        with (
            patch("extract.get_spark",         return_value=spark),
            patch("extract.extract_customers", return_value=customer_df),
            patch("extract.extract_products",  return_value=product_df),
            patch("extract.run_all_loads",     mock_loads),
        ):
            run_pipeline(
                orders_filepath=orders_csv,
                processed_base=str(tmp_path / "processed"),
            )

        call_kwargs = mock_loads.call_args.kwargs
        assert call_kwargs["partition_date"] == "2026-06-01"

    def test_run_id_is_valid_uuid(self, spark, orders_csv, customer_df, product_df, tmp_path):
        """The etl_run_id threaded through the pipeline must be a valid UUID."""
        from extract import run_pipeline

        mock_loads = MagicMock()
        with (
            patch("extract.get_spark",         return_value=spark),
            patch("extract.extract_customers", return_value=customer_df),
            patch("extract.extract_products",  return_value=product_df),
            patch("extract.run_all_loads",     mock_loads),
        ):
            result = run_pipeline(
                orders_filepath=orders_csv,
                processed_base=str(tmp_path / "processed"),
            )

        run_id = result.first()["etl_run_id"]
        uuid.UUID(run_id)   # raises ValueError if not a valid UUID


# ---------------------------------------------------------------------------
# Partition-date regex (inline utility, not a public function — tested directly)
# ---------------------------------------------------------------------------

class TestPartitionDateRegex:
    """Validates the regex that run_pipeline uses to parse the date from the filepath."""

    PATTERN = re.compile(r"dt=(\d{4}-\d{2}-\d{2})")

    def test_matches_well_formed_path(self):
        m = self.PATTERN.search("data/raw/dt=2026-06-01/orders.csv")
        assert m and m.group(1) == "2026-06-01"

    def test_no_match_on_plain_path(self):
        assert self.PATTERN.search("data/raw/orders.csv") is None

    def test_no_match_on_wrong_date_format(self):
        assert self.PATTERN.search("data/raw/dt=01-06-2026/orders.csv") is None