from kafka import KafkaProducer
import json
import time
from config import *

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def send_build_metadata():

    build_event = {
        "build_version": "1.0.0",
        "build_date": "2024-04-27T12:00:00Z",
        "commit_hash": "abcdef1234567890abcdef1234567890abcdef12",
        "build_number": "42",
        "builder": "AI-FraudProtection-Service CI CD Galaxy",
        "environment": "production",
        "project": "AI-FraudProtection-Service",
        "repository": "https://github.com/kartikgurnani/AI-FraudProtection-Service",
        "platform": "Windows",

        "cloud_providers": [
            {
                "provider": "Azure",
                "machine_type": "DCesv6-series Standard_DC128es_v6",
                "features": ["Intel® Trust Domain Extensions (TDX)"],
                "region": "East US",
                "cluster_zones": ["East US 1", "East US 2", "East US 3"],
                "autoscaling": {
                    "min_nodes": 3,
                    "max_nodes": 40,
                    "per_zone": True
                }
            },
            {
                "provider": "GCP",
                "machine_type": "c4d-standard-384",
                "additional_machine_type": "c4d-highmem-384",
                "region": "us-east1",
                "cluster_zones": ["us-east1-b", "us-east1-c", "us-east1-d"],
                "additional_cluster_zones": [
                    "us-central1-a",
                    "us-central1-b",
                    "us-central1-c"
                ],
                "autoscaling": {
                    "min_nodes": 3,
                    "max_nodes": 40,
                    "per_zone": True
                },
                "cluster_type": "3-node high-memory cluster"
            }
        ],

        "hardware": {
            "cpu": "Intel Xeon Platinum 8280",
            "cpu_cores": 28,
            "ram": "256GB",
            "storage": "2TB NVMe SSD",
            "os": "Windows Server 2022 Datacenter",
            "virtualization": "Cross-container service support",
            "additional_features": [
                "Hardware-accelerated security",
                "TPM 2.0"
            ]
        }
    }

    producer.send(TOPIC_NAME, build_event)
    producer.flush()

    print("✅ Build metadata event sent")


if __name__ == "__main__":
    while True:
        send_build_metadata()
        time.sleep(10)
