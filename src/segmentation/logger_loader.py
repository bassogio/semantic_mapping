import os
import logging

def is_running_in_container():
    """
    Determines if the code is running inside a container.
    """
    try:
        with open('/proc/self/cgroup', 'r') as file:
            if any('docker' in line or 'kubepods' in line for line in file):
                return True
    except FileNotFoundError:
        pass

    if os.path.exists('/run/.containerenv'):
        return True

    return False

def setup_logger():
    """
    Sets up the logger for the application.
    """
    log_dir = os.path.dirname(os.path.abspath(__file__))

    if is_running_in_container():
        log_path = os.path.join(log_dir, '../logs')
    else:
        log_path = os.path.join(log_dir, '../../logs/segmentation')

    os.makedirs(log_path, exist_ok=True)

    log_file = os.path.join(log_path, 'segmentation_logs.txt')

    with open(log_file, 'w'):
        pass

    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    logger = logging.getLogger('SegmentationPublisher')

    return logger

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Logger is set up successfully.")
    if is_running_in_container():
        print("The code is running inside a container.")
    else:
        print("The code is running on the host.")
