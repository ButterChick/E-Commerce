from faker import Faker
import psycopg2
import random
from decimal import Decimal

fake = Faker()

# PostgreSQL connection
conn = psycopg2.connect(
    host="localhost",
    database="ecommerce_oltp",
    user="postgres",
    password="admin",
    port="5432"
)

cursor = conn.cursor()

tiers = ["standard", "silver", "gold", "platinum"]

for _ in range(100):

    cursor.execute(
        """
        INSERT INTO customers (
            email,
            first_name,
            last_name,
            country,
            signup_date,
            customer_tier,
            is_active
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            fake.unique.email(),
            fake.first_name(),
            fake.last_name(),
            fake.country_code(),
            fake.date_between(start_date="-3y", end_date="today"),
            random.choice(tiers),
            random.choice([True, True, True, False])
        )
    )

# -----------------------------
# INSERT PRODUCTS
# -----------------------------

categories = [
    "electronics",
    "apparel",
    "books",
    "home",
    "sports"
]

for _ in range(50):

    unit_cost = round(random.uniform(5, 500), 2)

    list_price = round(
        unit_cost * random.uniform(1.1, 2.0),
        2
    )

    cursor.execute(
        """
        INSERT INTO products (
            sku,
            product_name,
            category,
            unit_cost,
            list_price,
            is_discontinued
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            fake.unique.bothify(text="SKU-#####"),
            fake.word().title(),
            random.choice(categories),
            Decimal(str(unit_cost)),
            Decimal(str(list_price)),
            random.choice([False, False, False, True])
        )
    )

# Save changes
conn.commit()

# Close connections
cursor.close()
conn.close()

print("Fake data inserted successfully.")