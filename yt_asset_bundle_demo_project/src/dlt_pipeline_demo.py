# Databricks notebook: 02_dlt_coffee_pipeline
# --------------------------------------------
import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

# ---------------- CONFIGURATION ----------------
CATALOG = spark.conf.get("demo.catalog", "retail")
SCHEMA  = spark.conf.get("demo.schema", "coffee_demo")
LANDING = spark.conf.get("demo.landing_dir", f"/Volumes/{CATALOG}/{SCHEMA}/coffee_landing/orders")
SYSTEM  = spark.conf.get("demo.system_dir",  f"/Volumes/{CATALOG}/{SCHEMA}/coffee_system")

STORE_TZ = "UTC"
CLOSE_HOUR_UTC = 21
OPEN_HOUR_UTC  = 6

# ---------------- BRONZE ------------------------
@dlt.table(
    name="orders_bronze",
    comment="Raw coffee orders from UC Volume via Auto Loader."
)
def orders_bronze():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", f"{SYSTEM}/schema/orders_bronze")
        .load(LANDING)
        .withColumn("ingest_time", current_timestamp())
    )

# ---------------- SILVER ------------------------
order_schema = StructType([
    StructField("order_id", StringType()),
    StructField("ts_utc",   StringType()),
    StructField("store_id", StringType()),
    StructField("city",     StringType()),
    StructField("sku",      StringType()),
    StructField("qty",      IntegerType()),
    StructField("unit_price", IntegerType()),
    StructField("amount",     IntegerType()),
    StructField("payment_type", StringType())
])

@dlt.view(name="orders_silver_view", comment="Typed orders with derived time columns.")
def orders_silver_view():
    df = dlt.read_stream("orders_bronze")
    cols = [f.name for f in order_schema]
    df = df.select([col(c).cast(order_schema[c].dataType) if c in cols else col(c) for c in df.columns])

    df = (
        df.withColumn("event_time", to_timestamp("ts_utc"))
          .withColumn("date_utc",   to_date("event_time"))
          .withColumn("hour_utc",   hour("event_time"))
          .withColumn("day_of_week", dayofweek("event_time"))  # ✅ Spark 3.x safe
          .withColumn("month", month("event_time"))
    )

    dlt.expect_or_drop("valid_amount", "amount >= 0")
    dlt.expect_or_drop("valid_qty",    "qty >= 1")
    dlt.expect_or_drop("valid_time",   "event_time IS NOT NULL")
    return df

@dlt.table(name="orders_silver", comment="Validated, typed streaming orders.")
def orders_silver():
    return dlt.read_stream("orders_silver_view")

# ---------------- GOLD: SALES KPIs ----------------
@dlt.table(name="store_sales_today_gold", comment="Per-store KPIs for current day.")
def store_sales_today_gold():
    df = (
        dlt.read_stream("orders_silver")
        .withWatermark("event_time", "10 minutes")  # ✅ added watermark
        .filter(col("date_utc") == current_date())
    )

    return (
        df.groupBy("store_id", "date_utc")
          .agg(
              sum("amount").alias("revenue"),
              approx_count_distinct("order_id").alias("orders"),  # ✅ fixed
              avg("amount").alias("avg_ticket")
          )
    )

@dlt.table(name="hourly_sales_gold", comment="Revenue per hour (current day).")
def hourly_sales_gold():
    df = (
        dlt.read_stream("orders_silver")
        .withWatermark("event_time", "10 minutes")  # ✅ added watermark
        .filter(col("date_utc") == current_date())
    )

    return (
        df.groupBy("date_utc", "hour_utc")
          .agg(
              sum("amount").alias("revenue_hr"),
              approx_count_distinct("order_id").alias("orders_hr")  # ✅ fixed
          )
    )

@dlt.table(name="product_sales_today_gold", comment="Top products by revenue today.")
def product_sales_today_gold():
    df = (
        dlt.read_stream("orders_silver")
        .withWatermark("event_time", "10 minutes")
        .filter(col("date_utc") == current_date())
    )
    return (
        df.groupBy("sku")
          .agg(sum("amount").alias("revenue"), sum("qty").alias("qty_sold"))
    )

@dlt.table(name="sales_summary_gold", comment="Latest snapshot for scorecards.")
def sales_summary_gold():
    df = dlt.read_stream("store_sales_today_gold")
    latest = (
        df.groupBy("date_utc")
          .agg(
              sum("revenue").alias("total_revenue"),
              sum("orders").alias("total_orders"),
              avg("avg_ticket").alias("avg_ticket")
          )
    )
    now_hr = hour(current_timestamp())
    return (
        latest
        .withColumn(
            "hours_since_open",
            when(lit(now_hr) > lit(OPEN_HOUR_UTC), lit(now_hr - OPEN_HOUR_UTC)).otherwise(lit(0))
        )
        .withColumn(
            "sales_rate_hr",
            when(col("hours_since_open") > 0, col("total_revenue") / col("hours_since_open")).otherwise(lit(0.0))
        )
    )

# ---------------- GOLD + AI PREDICTIONS ----------------
@dlt.table(name="sales_predictions_gold", comment="AI predictions: EOD revenue & seasonal baseline.")
def sales_predictions_gold():
    summary = dlt.read_stream("sales_summary_gold").alias("s")

    latest_time = (
        dlt.read("orders_silver")
        .select("event_time")
        .orderBy(desc("event_time"))
        .limit(1)
    )

    time_feats = (
        latest_time
        .withColumn("hour_of_day", hour("event_time"))
        .withColumn("day_of_week", dayofweek("event_time"))  # ✅ Spark 3.x safe
        .withColumn("month", month("event_time"))
        .select("hour_of_day", "day_of_week", "month")
    )

    base = summary.crossJoin(time_feats)

    # --- Coefficient tables aliased to avoid ambiguity ---
    eod_coeff = (
        spark.read.table(f"{CATALOG}.{SCHEMA}.ai_coeff_eod")
        .filter(col("model_type") == "EOD")
        .select(
            col("intercept").alias("eod_intercept"),
            col("coef_hour").alias("eod_coef_hour"),
            col("coef_dow").alias("eod_coef_dow"),
            col("coef_month").alias("eod_coef_month"),
            col("coef_rate").alias("eod_coef_rate")
        )
        .limit(1)
    )

    season_coeff = (
        spark.read.table(f"{CATALOG}.{SCHEMA}.ai_coeff_season")
        .filter(col("model_type") == "SEASON")
        .select(
            col("intercept").alias("season_intercept"),
            col("coef_month").alias("season_coef_month")
        )
        .limit(1)
    )

    # CrossJoin with static coefficients
    joined = base.crossJoin(eod_coeff).crossJoin(season_coeff)

    pred = (
        joined
        .withColumn(
            "pred_eod_revenue",
            col("eod_intercept")
            + col("eod_coef_hour") * col("hour_of_day")
            + col("eod_coef_dow") * col("day_of_week")
            + col("eod_coef_month") * col("month")
            + col("eod_coef_rate") * col("sales_rate_hr")
        )
        .withColumn(
            "pred_month_baseline",
            col("season_intercept") + col("season_coef_month") * col("month")
        )
        .select(
            current_date().alias("date_utc"),
            round(col("total_revenue"), 2).alias("total_revenue"),
            col("total_orders"),
            round(col("avg_ticket"), 2).alias("avg_ticket"),
            round(col("sales_rate_hr"), 2).alias("sales_rate_hr"),
            col("hour_of_day"),
            col("day_of_week"),
            col("month"),
            round(col("pred_eod_revenue"), 2).alias("pred_eod_revenue"),
            round(col("pred_month_baseline"), 2).alias("pred_month_baseline"),
        )
    )

    return pred
