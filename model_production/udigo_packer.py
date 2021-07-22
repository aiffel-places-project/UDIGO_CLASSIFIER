import os
import json
from udigo_service import UdigoPlaceService
from tensorflow.keras.models import load_model

model_path = os.listdir("./model")
assert "h5" in model_path[0]
model = load_model(f"./model/{model_path[0]}")

with open("./model/place_55_label.json", "r", encoding="utf-8-sig") as f:
    label_info = json.load(f)


def pack():
    udigo = UdigoPlaceService()
    udigo.pack("classifier", model)
    udigo.pack("label", label_info)

    saved_path = udigo.save()


if __name__ == "__main__":
    pack()
