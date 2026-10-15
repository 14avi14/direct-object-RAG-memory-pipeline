"""memory_creation.py - populates RAG memory DB

This takes in images, spatial coordinates, and timestamp data and
stores those in a RAG-based vector DB. Groups of images turned into a short
video clip which is sent to a VLM to be summarized. These summarizations
are embedded and stored for use during semantic similarity searches. The
associated timestamp and spatial, data are fed to the vector DB to be
stored as well(for details, check vector_store.py)

For the VLM, currently qwen 3.8 27b is being used and the API is Groq.
"""

import os
import random
import base64
import pprint
import time
from collections import Counter

import numpy as np
import cv2
import requests
from dotenv import load_dotenv


from vision import get_positions_from_video_frames

load_dotenv()


USE_LLM = True

MAX_FRAMES_PER = 10

URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY_REF = os.getenv("GROQ_API_KEY")
headers = {
    "Authorization": f"Bearer {API_KEY_REF}",
    "Content-Type": "application/json"
}
LLM_MODEL_NAME = "qwen/qwen3.8-27b"

CAPTIONING_PROMPT = """You are an expert navigation assistant. Analyze the
image and generate a 2 to 3 sentence detailed description of the primary
object, specifically tailored for a visual RAG navigation
database. Focus on distinctive visual markers, color, shape, texture,
logos/text, and spatial context that would help a traveler or autonomous
system relocate this landmark or object.

If the image is too occluded or blurry, DON'T OUTPUT ANYTHING(blank string, NO
REASONING).
"""

def encode_to_base64(obj):
    return base64.b64encode(obj).decode("utf-8")

def get_object_description(image_url):
    if not USE_LLM:
        # return mock caption
        return """A person with a blue jacket, red pants, and dark hair,
                holding a yellow pencil"""

    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": CAPTIONING_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 900
    }

    print("[memory_creation.py - get_caption(vid)]")
    response = requests.post(url=URL, headers=headers, json=payload)
    
    response_json = response.json()
    if "error" in response_json:
        print("-" * 25 + " API Response Error " + "-" * 25)
        text = None
        pprint.pprint(response_json)
    else:
        text = response_json["choices"][0]["message"]["content"]
        
    return text

def memorize(frames, depth_frames, timestamps, vec_db):
    for i in range(0, len(frames), MAX_FRAMES_PER):
        # Get object detections from video

        # TIME FOR DEBUGGING PURPOSES
        start = time.perf_counter()
        print("Starting vision processing")
        detections, class_names = get_positions_from_video_frames(frames[i:i+MAX_FRAMES_PER], depth_frames[i:i+MAX_FRAMES_PER])
        print(f"Total time for vision processing: {time.perf_counter() - start}\n")

        # Put the objects into the vector database
        for key, detection in detections.items():
            img = detection["cutout"]
            position = detection["pos"]
            times_seen = 1

            # For testing: visualization
            cv2.imshow("Cutout", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # If image is too small, then it will be indiscernible
            min_w, min_h = 48, 48
            if len(img) < min_h or len(img[0]) < min_w:
                print(f"Image was less than {min_w}x{min_h} pixels. Could not be processed")
                continue

            # NumPy to base 64
            byte_encoded = cv2.imencode(".jpg", img)[1].tobytes() 
            b64_encoded = encode_to_base64(byte_encoded)
            image_url = f"data:image/jpeg;base64,{b64_encoded}"
            
            # Feed to VLM to describe object
            try:
                detailed_descr = get_object_description(image_url)
                while detailed_descr is None:
                    print("Trying again after 5 seconds. Press CTRL+C to stop.")
                    print("Data already processed will still be saved.\n")
                    time.sleep(5)
                    detailed_descr = get_object_description(image_url)
            except KeyboardInterrupt:
                print("Stopping memory processing\n")
                return

            if not detailed_descr: # Continue if blank string
                print("Image was too occluded")
                continue

            # Check if already exists in memory
            docs = vec_db.euclidian_similarity_query(detection["pos"], "location", top_k=5)

            similar_obj_id = None
            dist_threshold = 0.3
            similarity_threshold = 0.85
            for doc in docs:
                if doc["distance"] < 0.3:
                    other_descr = doc["data"]["text"]
                    cosine_similarity = vec_db.get_similarity(other_descr, detailed_descr)
                    
                    if cosine_similarity >= similarity_threshold:
                        # Found what is likely the same object
                        times_seen = doc["data"]["times_seen"] + 1
                        position = (doc["data"]["location"] * doc["data"]["times_seen"] + position) / times_seen # Make new position running average of old positions
                        similar_obj_id = doc["id"]
                        break
            
            # Load into memory
            timestamp_str =  time.strftime("%H:%M:%S %D", time.localtime(timestamps[i]))
            doc = {
                "text": detailed_descr, "location": position,
                "last_timestamp": timestamps[i], "last_timestamp_str": timestamp_str,
                "times_seen": times_seen
            }

            print("FINAL INSERT:")
            pprint.pprint(doc)

            if similar_obj_id is None:
                vec_db.insert([doc])
            else:
                vec_db.update(similar_obj_id, doc)

        if os.path.isfile(temp_file_name):
            os.remove(temp_file_name) # Not sure if this will help with latency at all



