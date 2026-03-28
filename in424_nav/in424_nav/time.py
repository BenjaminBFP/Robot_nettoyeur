import json
import numpy as np
from time import time


path = "./ros2_ws/src/IN424/in424_nav/in424_nav/log_time.json"
with open(path, "r") as f:
    donnees = json.load(f)
print(f"max(donnees): {max(donnees)}")
print(f"moyenne(donnees): {np.mean(donnees)}")
