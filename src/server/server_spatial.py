import base64
import json
import logging
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import json_numpy
json_numpy.patch()

import draccus
from dataclasses import dataclass
import uvicorn
import os
import json

# ------------------------------------------
# Spatial Inference Server
# ------------------------------------------
class SpatialServer:
    def __init__(self, model) -> None:
        logging.info("Initializing SpatialVla model...")
        self.model = model
        self.task_description = "put the spoon on the towel"
        self.model.reset(self.task_description)  # Pre-warm model
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        self.app.post("/reset")(self.reset_model)
        self.app.post("/set_task")(self.set_task_model)
        
    async def set_task_model(self, task_description: str):
        self.task_description = task_description
        self.model.reset(self.task_description)
    async def reset_model(self):
        self.model.reset(self.task_description)
    
    async def predict_action(self, request: Request) -> JSONResponse:
        try:
            payload = await request.json()  # <-- await here
            instruction = payload["instruction"]
            new_image = np.array(payload["image"], dtype=np.uint8)
            raw_action, action = self.model.step(new_image, instruction)
            return JSONResponse(content={
                "raw_action": {k: v.tolist() for k, v in raw_action.items()},
                "action": {k: v.tolist() for k, v in action.items()},
            })
        except Exception as e:
            logging.error(traceback.format_exc())
            return JSONResponse(content={"error": str(e)}, status_code=500)


    def run(self, host: str = "0.0.0.0", port: int = 9020) -> None:
        import uvicorn
        logging.info(f"🚀 Running SpatialVla server on http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(current_dir, "../../model_config.json"))
    saved_model_path = json.load(open(config_path, "r"))["spatial_path"] 
    
    model = SpatialVLAInference(
        saved_model_path=saved_model_path,
        policy_setup="widowx_bridge",
        action_scale=1,
    )
    server = SpatialServer(model)
    server.run()
