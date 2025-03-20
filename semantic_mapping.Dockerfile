# Use Dusty-NV's container as a base
FROM dustynv/ros:humble-ros-base-l4t-r36.3.0

RUN apt-get update && apt-get install -y \
    python3-pip \
    usbutils 

# Copy the requirements.txt file into the container
COPY requirements.txt /workspace/

# Install Python libraries using requirements.txt
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Set the working directory inside the container
WORKDIR /workspace

# Copy necessary application files to the container
COPY src/ /workspace/src/
COPY config/ /workspace/config/

RUN chmod +x /workspace/src/entrypoint.sh

CMD ["/bin/bash"]

