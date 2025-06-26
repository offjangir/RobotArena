import numpy as np
import genesis as gs
import cv2
from transforms3d.euler import euler2quat, quat2euler
import os
import yaml
import json
import requests
import json_numpy
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import base64


class OctoAgent:
    def __init__(self, url):
        self.url = url

    def decode_action(self, action_str):
        # Parse the JSON string into a Python dictionary
        # action = json.loads(action_str)
        action = {}
        # Decode each Base64-encoded NumPy array
        for key, value in action_str.items():
            if isinstance(value, dict) and "__numpy__" in value:
                # Decode the Base64 string to bytes
                decoded_data = base64.b64decode(value["__numpy__"])
                # Convert the decoded bytes into a NumPy array using the dtype and shape
                dtype = np.dtype(value["dtype"])
                shape = tuple(value["shape"])
                action[key] = np.frombuffer(decoded_data, dtype=dtype).reshape(shape)

        return action
    
    def get_action(self, image: np.ndarray, instruction: str):
        # Encode image as PNG and then base64 for JSON serialization
        
        success, encoded_image = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            print("Failed to encode image")
            return None

        image_b64 = base64.b64encode(encoded_image).decode("utf-8")
        
        payload = {
            "image": image_b64,
            "instruction": instruction,
        }
        try:
            response = requests.post(self.url, json=payload)
        except Exception as e:
            print("Error calling Octo API:", e)
            return None

        if response.status_code != 200:
            print("API error:", response.text)
            return None

        result = response.json()
        # Decode the action

        raw_action = self.decode_action(result.get("raw_action"))
        action = self.decode_action(result.get("action"))
        # print("Received action:", raw_action, "and", action)
        return raw_action, action
