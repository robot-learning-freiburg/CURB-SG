#!/bin/bash
source /home/greve/PycharmProjects/collaborative_panoptic_mapping/catkin_ws/devel/setup.bash

# Run the command and save its PID
"$@" &
PID=$!

# Define a function to handle SIGINT
function handle_sigint {
    kill -SIGINT "$PID"
}

# Set the function to handle SIGINT
trap handle_sigint SIGINT

# Wait for the command to finish
wait "$PID"
