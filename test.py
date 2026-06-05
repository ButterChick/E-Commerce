import os
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = os.environ["PATH"] + r";C:\hadoop\bin"
os.environ["PYSPARK_PYTHON"] = r"C:\Users\DELL\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\DELL\AppData\Local\Programs\Python\Python311\python.exe"

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()

df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "value"])
df.show()

print("Spark is working correctly")
spark.stop()