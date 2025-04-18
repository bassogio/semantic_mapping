# Use Dusty-NV's container as a base
FROM dustynv/ros:humble-ros-base-l4t-r36.3.0

RUN apt-get update && apt-get install -y \
    python3-pip \
    usbutils \
    libopenblas-base \
    libopenmpi-dev \
    ros-humble-sensor-msgs-py \
    ros-humble-ros2bag

# Clone the torchvision repository and install from source
RUN git clone https://github.com/pytorch/vision.git && \
    cd vision && pip install -e . && cd ..

# Copy the requirements.txt file into the container
COPY requirements.txt /workspace/

# Install Python libraries using requirements.txt
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Set the working directory inside the container
WORKDIR /workspace

# Copy necessary application files to the container
# COPY src/ /workspace/src/
# COPY config/ /workspace/config/
COPY entrypoint.sh /workspace/

RUN chmod +x /workspace/entrypoint.sh

# Set up the ROS environment to be sourced on each new shell session
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc


CMD ["/bin/bash"]

