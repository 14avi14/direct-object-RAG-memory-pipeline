"""run_pipeline.py - runs the full RAG based memory pipeline.

Recieves raw sensor data from the car and either stores in memory via
VectorStore class from vector_store.py, or uses the data in addition to
already stored memory to answer a query entered by the user. 
"""


import socket
import json
import pickle
import pprint
import time
import sys
import os

import socket_ops
from vector_store import VectorStore
from memory_creation import memorize
from robot_action_planner import obtain_context, get_final_position

MEMORY_FILE_NAME = "vec_db.pkl"

# Step 0. Establish connection
print("Establishing connection...")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(socket_ops.SERVER_ADDRESS)
print(socket_ops.SERVER_ADDRESS)
server.listen(1)

(conn, addr) = server.accept()

print(f"CONNECTION: {addr}")

mode = input("Choose a mode(MEMORY or INFERENCE): ")


if mode.upper() == "MEMORY":
    frames = []
    depth_frames = []
    car_states = []
    timestamps = []
    skip = 1 # 1 image will be stored every `skip` number of frames
elif mode.upper() == "INFERENCE":
    pass
else:
    print("INVALID MODE")
    sys.exit()

if os.path.isfile(MEMORY_FILE_NAME):
    with open(MEMORY_FILE_NAME, "rb") as f:
        vec_db = pickle.load(f)
else:
    vec_db = VectorStore()


print("Memory Preview")
pprint.pprint(vec_db.docs[:5])

iterations = 0

while True:

    if iterations % 10 == 0:
        run = input("Would you like to continue(y/n)?: ")
        if run.lower() == "n":
            break
    
    if mode == "MEMORY":
        # Step 1.
        print("----- RECIEVING DATA -----")
        bgr_image = socket_ops.recieve_image(conn)
        depth_image = socket_ops.recieve_image(conn)
        car_state = json.loads(socket_ops.recieve_string(conn))
        timestamp = round(float(socket_ops.recieve_string(conn)))
        print("----- DATA RECIEVED -----")

        if iterations % skip == 0:
            frames.append(bgr_image)
            depth_frames.append(depth_image)
            car_states.append(car_state)
            timestamps.append(timestamp)

        # Step 2.
        return_msg = {"goal_pose": None, "reasoning": "In memory mode."}

        socket_ops.send_string(conn, json.dumps(return_msg))

    if mode == "INFERENCE":
        query = input("Please input a query for INFERENCE mode: ")

        # Step 1.
        print("----- RECIEVING DATA -----")
        image = socket_ops.recieve_image(conn)
        car_state = json.loads(socket_ops.recieve_string(conn))
        timestamp = round(float(socket_ops.recieve_string(conn)))

        loc = car_state["position_xy"]
        heading = car_state["heading_deg"]

        # Step 2.        
        print("----- COMBINING WITH MEMORY DATA -----")
        current_context = {"image": image, "location": loc, "heading_deg": heading, "timestamp": timestamp}
        final_context = obtain_context(query, vec_db, current_context=current_context)
        final_json = get_final_position(query, final_context)

        socket_ops.send_string(conn, json.dumps(final_json))
        
        # Step 3.
        print("----- FINAL -----")
        pprint.pprint(final_json)

    iterations += 1

print("----- SERVER CONNECTION SHUTTING DOWN -----")
conn.shutdown(socket.SHUT_RDWR)
conn.close()
server.close()

if mode == "MEMORY":
    save_memory = input("Would you like to save the memory(y/n)?: ")
    if save_memory.lower() == "y":
        print("----- SAVING TO MEMORY -----")
        start = time.time()
        print(f"# Frames: {len(frames)}")
        memorize(frames, depth_frames, timestamps, vec_db)
        with open(MEMORY_FILE_NAME, "wb") as f:
            pickle.dump(vec_db, f, -1)
        print()
        print(25 * "-" + "Memory saved to " + MEMORY_FILE_NAME + 25 * "-")
        print(f"Took {time.time() - start} seconds")
