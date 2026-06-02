import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="ecommerce_oltp",
    user="postgres",
    password="admin",
    port="5432"
)

