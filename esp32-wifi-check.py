import requests
ESP32_IP = "172.20.10.6"

requests.get(
    f"http://{ESP32_IP}/send",
    params={"a": 1000, "b": 2000, "c": 1500}
)
