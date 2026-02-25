from kafka import KafkaProducer
import json
from kafka_config import *

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    acks="all",
    enable_idempotence=True,
    value_serializer=lambda v: json.dumps(v).encode()
)

event = {
    "build_version": "1.0.0",
    "build_date": "2024-04-27",
    "commit_hash": "abcdef123",
    "environment": "production",
    "project": "AI-FraudProtection-Service",
    "platform": "Windows"
}

producer.send(TOPICS["build"], event)
producer.flush()
print("Build event sent")
