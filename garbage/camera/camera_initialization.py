#camera_initialization
import pyrealsense2 as rs

class CameraInit:
    def __init__(self):
        """
        Initializes the RealSenseCamera object by loading configuration, 
        setting up the RealSense pipeline, and enabling the required streams.
        """
        # Initialize the RealSense pipeline and configuration
        self.pipeline = rs.pipeline()
        self.realsense_config = rs.config()

        # Find the available RealSense device
        self.device = self.find_device()

        # Retrieve the stream configurations dynamically
        self.color_stream_config, self.depth_stream_config = self.get_stream_configurations()

        # Enable the streams using the retrieved configurations
        self.realsense_config.enable_stream(
            rs.stream.color, 
            self.color_stream_config['width'], 
            self.color_stream_config['height'], 
            self.color_stream_config['format'], 
            self.color_stream_config['fps']
        )
        self.realsense_config.enable_stream(
            rs.stream.depth, 
            self.depth_stream_config['width'], 
            self.depth_stream_config['height'], 
            self.depth_stream_config['format'], 
            self.depth_stream_config['fps']
        )

        # Start the pipeline
        self.pipeline.start(self.realsense_config)
        #logger.info("RealSense pipeline started with dynamic stream configurations.")

    def find_device(self):
        """
        Searches for RealSense devices and initializes the first available device.
        Raises an error if no devices are found.
        """
        #logger.info("Searching for RealSense devices...")
        try:
            ctx = rs.context()
            devices = ctx.query_devices()

            # Check if any devices are found
            if not devices:
                #logger.error("No RealSense devices found!")
                raise RuntimeError("No RealSense devices found!")

            # Select the first device
            device = devices[0]
            #logger.info(f"Found device: {device.get_info(rs.camera_info.name)}")
            return device
        except RuntimeError as e:
            #logger.error(f"Error finding device: {e}")
            raise

    def get_stream_configurations(self):
        """
        Retrieves the configurations for color and depth streams dynamically from the device.
        """
        #logger.info("Retrieving stream configurations...")
        color_stream_config = None
        depth_stream_config = None

        try:
            # Use a temporary pipeline to query the supported stream profiles
            temp_pipeline = rs.pipeline()
            profile = temp_pipeline.start()
            sensor = profile.get_device().query_sensors()

            for s in sensor:
                for stream_profile in s.get_stream_profiles():
                    if stream_profile.stream_type() == rs.stream.color:
                        video_profile = stream_profile.as_video_stream_profile()
                        color_stream_config = {
                            'width': video_profile.width(),
                            'height': video_profile.height(),
                            'format': video_profile.format(),
                            'fps': video_profile.fps()
                        }
                        #logger.info(f"Color stream config: {color_stream_config}")
                    elif stream_profile.stream_type() == rs.stream.depth:
                        video_profile = stream_profile.as_video_stream_profile()
                        depth_stream_config = {
                            'width': video_profile.width(),
                            'height': video_profile.height(),
                            'format': video_profile.format(),
                            'fps': video_profile.fps()
                        }
                        #logger.info(f"Depth stream config: {depth_stream_config}")

            # Stop the temporary pipeline
            temp_pipeline.stop()

        except Exception as e:
            #logger.error(f"Error retrieving stream configurations: {e}")
            raise

        if not color_stream_config or not depth_stream_config:
            raise RuntimeError("Failed to retrieve stream configurations.")
        
        return color_stream_config, depth_stream_config

    def release(self):
        """
        Stops the RealSense pipeline and releases resources.
        """
        self.pipeline.stop()
        #logger.info("RealSense pipeline stopped.")
