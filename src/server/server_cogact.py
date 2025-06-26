import os
import torch
import numpy as np
import tensorflow as tf
import argparse
import numpy as np
import base64
import json
import logging
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import json_numpy
json_numpy.patch()

import draccus
from dataclasses import dataclass
import uvicorn


# ------------------------------------------
# Cogact Inference Server
# ------------------------------------------
class CogactServer:
    def __init__(self, model) -> None:
        logging.info("Initializing Cogact model...")
        self.model = model
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


    def run(self, host: str = "0.0.0.0", port: int = 9030) -> None:
        import uvicorn
        logging.info(f"🚀 Running Cogact server on http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from simpler_env.policies.sim_cogact import CogACTInference
    model = CogACTInference(
        saved_model_path='CogACT/CogACT-Base',  # e.g., CogACT/CogACT-Base
        policy_setup='widowx_bridge',
        action_scale=1,
        action_model_type='DiT-B',
        cfg_scale=1.5                     # cfg from 1.5 to 7 also performs well
    )  
    server = CogactServer(model)
    server.run()