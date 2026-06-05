from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import LongType
import uuid
from typing import Optional
import os
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = os.environ["PATH"] + r";C:\hadoop\bin"
os.environ["PYSPARK_PYTHON"] = r"C:\Users\DELL\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\DELL\AppData\Local\Programs\Python\Python311\python.exe"


# -------------------------
# SPARK SESSION
# -------------------------

def get_spark(app_name: str = "ecommerce_etl") -> SparkSession:
    """
    Get or create a local SparkSession.
    In production, remove the master("local[*]") line and let the
    cluster config inject it via spark-submit.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.jars", r"C:\Program Files\PostgreSQL\postgresql-42.7.11.jar")
        .getOrCreate()
    )


# -------------------------
# EXTRACT
# -------------------------

def extract_orders(spark: SparkSession, filepath: str) -> DataFrame:
    """
    Read orders CSV and cast ID columns to LongType (nullable).

    Args:
        spark: Active SparkSession.
        filepath: Path to the orders CSV file.

    Returns:
        Spark DataFrame with customer_id, product_id, order_id as Long.
    """
    df = spark.read.csv(filepath, header=True, inferSchema=False)

    for col in ("customer_id", "product_id", "order_id"):
        df = df.withColumn(col, F.col(col).cast("double").cast(LongType()))

    df = df.withColumn("quantity", F.col("quantity").cast("integer"))
    df = df.withColumn("unit_price", F.col("unit_price").cast("double"))

    return df

def extract_customers(spark: SparkSession, jdbc_url: str, jdbc_props: dict) -> DataFrame:
    """
    Pull customer_id, customer_tier, and country from the OLTP DB via JDBC.
    Args:
        spark:      Active SparkSession.
        jdbc_url:   JDBC connection string.
        jdbc_props: Dict with 'user', 'password', 'driver'.
    Returns:
        Spark DataFrame with columns: customer_id, customer_tier, country.
    """
    return spark.read.jdbc(
        url=jdbc_url,
        table="(SELECT customer_id, customer_tier, country FROM customers) t",
        properties=jdbc_props,
    )

def extract_products(spark: SparkSession, jdbc_url: str, jdbc_props: dict) -> DataFrame:
    """
    Pull product_id, category, and unit_cost from the OLTP DB via JDBC.
    Args:
        spark:      Active SparkSession.
        jdbc_url:   JDBC connection string.
        jdbc_props: Dict with 'user', 'password', 'driver'.

    Returns:
        Spark DataFrame with columns: product_id, category, unit_cost.
    """
    return spark.read.jdbc(
        url=jdbc_url,
        table="(SELECT product_id, category, unit_cost FROM products) t",
        properties=jdbc_props,
    )
# -------------------------
# TRANSFORM
# -------------------------
def transform(
    orders: DataFrame,
    customers: DataFrame,
    products: DataFrame,
    etl_run_id: Optional[str] = None,
) -> DataFrame:
    """
    Join orders with customers and products, compute revenue/margin fields,
    and enforce a final column order.
    Args:
        orders:      Output of extract_orders().
        customers:   Output of extract_customers().
        products:    Output of extract_products().
        etl_run_id:  UUID string to tag the batch. Auto-generated if None.

    Returns:
        Spark DataFrame ready for loading.
    """
    run_id = etl_run_id if etl_run_id is not None else str(uuid.uuid4())

    fact = (
        orders
        .join(customers, on="customer_id", how="left")
        .join(products, on="product_id", how="left")
        .withColumnRenamed("country", "customer_country")
        .withColumnRenamed("category", "product_category")
        .withColumn(
            "gross_revenue",
            F.round(F.col("quantity") * F.col("unit_price"), 2)
        )
        .withColumn(
            "gross_margin",
            F.round(F.col("quantity") * (F.col("unit_price") - F.col("unit_cost")), 2)
        )
        .withColumn(
            "margin_pct",
            F.when(
                F.col("gross_revenue") == 0,
                F.lit(None).cast("double")
            ).otherwise(
                F.round((F.col("gross_margin") / F.col("gross_revenue")) * 100, 2)
            )
        )
        .withColumn("etl_run_id", F.lit(run_id))
        .select(
            "order_item_id",
            "order_id",
            "order_date",
            "customer_id",
            "customer_tier",
            "customer_country",
            "product_id",
            "product_category",
            "quantity",
            "unit_price",
            "unit_cost",
            "gross_revenue",
            "gross_margin",
            "margin_pct",
            "channel",
            "etl_run_id",
        )
    )

    return fact
# -------------------------
# DIAGNOSTICS
# -------------------------
def log_quality(fact: DataFrame) -> None:
    """Print basic data quality counts to stdout."""
    total = fact.count()
    missing_customer = fact.filter(F.col("customer_tier").isNull()).count()
    missing_product = fact.filter(F.col("product_category").isNull()).count()

    print(f"\nDataset shape: {total} rows x {len(fact.columns)} columns")
    print(f"Missing customer matches: {missing_customer}")
    print(f"Missing product matches:  {missing_product}")
    fact.show(5, truncate=False)

# -------------------------
# ORCHESTRATOR
# -------------------------
def run_pipeline(
    orders_filepath: str,
    oltp_host: str = "localhost",
    oltp_db: str = "ecommerce_oltp",
    oltp_user: str = "postgres",
    oltp_password: str = "admin",
    oltp_port: str = "5432",
) -> DataFrame:
    """
    Full ETL run: extract from CSV + OLTP DB, transform, return fact DataFrame.
    Args:
        orders_filepath: Path to the orders CSV.
        oltp_*:          Connection parameters for the OLTP Postgres database.
    Returns:
        Transformed Spark DataFrame.
    """
    spark = get_spark()
    jdbc_url = f"jdbc:postgresql://{oltp_host}:{oltp_port}/{oltp_db}"
    jdbc_props = {
        "user": oltp_user,
        "password": oltp_password,
        "driver": "org.postgresql.Driver",
    }

    orders = extract_orders(spark, orders_filepath)
    customers = extract_customers(spark, jdbc_url, jdbc_props)
    products = extract_products(spark, jdbc_url, jdbc_props)

    fact = transform(orders, customers, products)
    log_quality(fact)

    return fact
# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    run_pipeline("data/raw/dt=2026-06-01/orders.csv")