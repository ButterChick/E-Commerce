import psycopg2
import pandas as pd
import random
from faker import Faker
from pathlib import Path

# ---------------------------
# CONFIG
# ---------------------------

DB_CONFIG = {
    "host": "localhost",
    "database": "ecommerce_oltp",
    "user": "postgres",
    "password": "admin",
    "port": "5432"
}

NUM_ROWS = 5000
PROCESS_DATE = "2026-06-01"

fake = Faker()

# ---------------------------
# LOAD CUSTOMERS & PRODUCTS
# ---------------------------

conn = psycopg2.connect(**DB_CONFIG)

customers = pd.read_sql(
    "SELECT customer_id FROM customers",
    conn
)

products = pd.read_sql(
    """
    SELECT
        product_id,
        list_price
    FROM products
    """,
    conn
)

conn.close()

customer_ids = customers["customer_id"].tolist()

products_data = products.to_dict("records")

# ---------------------------
# GENERATE ORDERS
# ---------------------------

channels = ["web", "mobile", "store"]

rows = []

for order_item_id in range(1, NUM_ROWS + 1):

    product = random.choice(products_data)

    product_id = product["product_id"]

    list_price = float(product["list_price"])

    # Multiple items per order
    order_id = random.randint(
        1,
        NUM_ROWS // 3
    )

    customer_id = random.choice(
        customer_ids
    )

    quantity = random.randint(
        1,
        10
    )

    # Actual selling price
    unit_price = round(
        list_price * random.uniform(
            0.8,
            1.05
        ),
        2
    )

    channel = random.choice(
        channels
    )

    row = {
        "order_item_id": order_item_id,
        "order_id": order_id,
        "order_date": PROCESS_DATE,
        "customer_id": customer_id,
        "product_id": product_id,
        "quantity": quantity,
        "unit_price": unit_price,
        "channel": channel
    }

    # ---------------------------
    # DATA QUALITY ISSUES
    # ---------------------------

    # Negative quantity (~1%)
    if random.random() < 0.01:
        row["quantity"] *= -1

    # Zero or negative price (~1%)
    if random.random() < 0.01:
        row["unit_price"] = random.choice([
            0,
            -round(
                random.uniform(
                    1,
                    100
                ),
                2
            )
        ])

    # Missing customer_id (~0.5%)
    if random.random() < 0.005:
        row["customer_id"] = None

    # Missing product_id (~0.5%)
    if random.random() < 0.005:
        row["product_id"] = None

    rows.append(row)

# ---------------------------
# DUPLICATE ORDER ITEM IDS
# ---------------------------

duplicate_count = int(
    NUM_ROWS * 0.01
)

duplicates = random.sample(
    rows,
    duplicate_count
)

for row in duplicates:

    duplicate_row = row.copy()

    rows.append(
        duplicate_row
    )

# ---------------------------
# SAVE CSV
# ---------------------------

df = pd.DataFrame(rows)

output_dir = Path(
    f"data={PROCESS_DATE}"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    output_dir / "orders.csv"
)

df.to_csv(
    output_file,
    index=False
)

print(
    f"Generated {len(df):,} rows"
)

print(
    f"Saved to {output_file}"
)

print(
    f"Unique orders: {df['order_id'].nunique():,}"
)