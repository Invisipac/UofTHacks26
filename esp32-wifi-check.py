# import requests
# ESP32_IP = "172.20.10.5"

# requests.get(
#     f"http://{ESP32_IP}/send",
#     params={"a": 1000, "b": 2000, "c": 1500}
# )

import requests

ESP32_HOST = "esp32-mixer.local"

r = requests.get(
    f"http://{ESP32_HOST}/send",
    params={"a": 1000, "b": 2000, "c": 1500},
    timeout=5
)

print(r.text)

