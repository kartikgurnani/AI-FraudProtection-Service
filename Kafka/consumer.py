from kafka import KafkaConsumer
import json
from config import *

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("🚀 Listening for build metadata events...\n")

def process_event(event):

    print("📦 Build Event Received")
    print("-" * 50)

    print("Version:", event.get("build_version"))
    print("Environment:", event.get("environment"))
    print("Project:", event.get("project"))
    print("Commit:", event.get("commit_hash"))
    print("Platform:", event.get("platform"))

    # Cloud info
    providers = event.get("cloud_providers", [])
    for provider in providers:
        print(f"\n☁ Cloud Provider: {provider.get('provider')}")
        print("Region:", provider.get("region"))
        print("Machine:", provider.get("machine_type"))

    # Hardware info
    hardware = event.get("hardware", {})
    print("\n🖥 Hardware:")
    print("CPU:", hardware.get("cpu"))
    print("RAM:", hardware.get("ram"))

    print("\n✅ Event processed\n")


for message in consumer:
    process_event(message.value)
