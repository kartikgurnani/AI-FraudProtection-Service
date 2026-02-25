from kafka import KafkaConsumer, KafkaProducer
import json
from kafka_config import *

consumer = KafkaConsumer(
    TOPICS["fraud_version"],
    bootstrap_servers=KAFKA_BROKER,
    enable_auto_commit=False,
    value_deserializer=lambda x: json.loads(x.decode())
)

dlq_producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode()
)

for msg in consumer:
    try:
        data = msg.value
        print("Processing model version:", data["model_version"])

        # process logic here

        consumer.commit()

    except Exception as e:
        print("Error → sending to DLQ")
        dlq_producer.send(TOPICS["dlq"], msg.value)
