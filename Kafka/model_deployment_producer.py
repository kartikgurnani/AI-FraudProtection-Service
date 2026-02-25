from kafka import KafkaProducer
import json
from kafka_config import *

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    acks="all",
    enable_idempotence=True,
    value_serializer=lambda v: json.dumps(v).encode()
)

deployment_event = {
    "model_name": "fraud_detector",
    "version": "v3",
    "accuracy": 0.98,
    "deployed_at": "2026-02-26",
    "environment": "production"
}

producer.send(TOPICS["deployment"], deployment_event)
producer.flush()
