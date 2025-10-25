# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT count(*) FROM retail.coffee_demo.ai_coeff_eod group by model_type HAVING count(*) > 1