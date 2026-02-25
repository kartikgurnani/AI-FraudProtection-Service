import streamlit as st
from kafka import KafkaConsumer
import json

st.title("🚀 Build Events Dashboard")

consumer = KafkaConsumer(
    "build_metadata",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode())
)

placeholder = st.empty()

for msg in consumer:
    placeholder.json(msg.value)
