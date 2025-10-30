from setuptools import setup, find_packages

setup(
    name="permafrost_dt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pika",  # For RabbitMQ
        "influxdb-client",  # For InfluxDB
        "pytest",  # For testing
    ],
    python_requires=">=3.7",
)