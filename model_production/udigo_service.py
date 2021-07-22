import os
import random
from typing import List

import numpy as np
import cv2
from bentoml import api, artifacts, env, BentoService
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.service.artifacts.common import JSONArtifact
from bentoml.adapters import ImageInput


@env(pip_packages=["tensorflow-cpu==2.5", "numpy", "opencv-python"])
@artifacts([KerasModelArtifact("classifier"), JSONArtifact("label")])
class UdigoPlaceService(BentoService):
    @api(input=ImageInput(), batch=False)
    def predict(self, img: List[np.ndarray]) -> List[str]:
        label_info = self.artifacts.label
        img = cv2.resize(img, (224, 224))
        image = img[np.newaxis, :, :, :]
        pred = self.artifacts.classifier.predict(image)
        class_index = str(np.argmax(pred))
        sen = random.choice(label_info[class_index]["sentence"])
        return {"name": label_info[class_index]["category"], "sentence": sen}
