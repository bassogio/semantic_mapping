# semantic_mapping

## Table of Contents

- [Setup and Deployment](#setup-and-deployment)
- [Project Structure](#project-structure)
- [Logging](#logging)
- [Utilities](#utilities)
- [Scripts](#scripts)
- [Troubleshooting](#troubleshooting)

---
## Setup and Deployment

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/bassogio/semantic_mapping.git
   cd your-project
   
### Prerequisites
Before setting up the project, ensure you have the following installed:

#### **If running without a container:**
- Python 3.8.10 and pip  
- Required dependencies (listed in `requirements.txt`)  
- ROS2 

#### **If running using a container:**
- Docker  

---

## Project Structure

```
├── config
│   ├── camera_config.yaml
│   ├── point_cloud_config.yaml
│   └── README.md
├── entrypoint.sh
├── README.md
├── requirements.txt
├── scripts
│   ├── cleanup.sh
│   ├── setup.sh
│   ├── shutdown.sh
│   └── start.sh
├── semantic_mapping.Dockerfile
├── src
│   ├── camera
│   └── point_cloud
└── ToDo.txt
```

---

### Key Directories

- **config/**: Contains configuration files for camera and point cloud processing. You may need to adjust the `point_cloud_config.yaml` file based on your system setup.
- **scripts/**: Includes shell scripts for managing the container (setup, start, shutdown, cleanup).
- **src/**: Contains source code for various modules (camera, point cloud, costmap, etc.).

---

## Managing the System

You can run the following scripts from the `scripts/` directory:

- `setup.sh`: Sets up the environment and dependencies.
- `start.sh`: Starts the Docker container.
- `shutdown.sh`: Stops all running containers.
- `cleanup.sh`: Removes all Docker containers.

---


