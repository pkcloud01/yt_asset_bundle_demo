from setuptools import setup, find_packages

setup(
    name="demo_databricks_asset_bundle",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "delta-spark",
        "pyspark"
    ]
)