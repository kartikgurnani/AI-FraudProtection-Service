from kafka import KafkaConsumer
import json
from kafka_config import *

consumer = KafkaConsumer(
    TOPICS["dlq"],
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda x: json.loads(x.decode())
)

print("Listening DLQ...")

for msg in consumer:
    print("Failed message:", msg.value)
