# Used the following site as a referance: https://huggingface.co/blog/clipseg-zero-shot
# https://huggingface.co/docs/transformers/en/index

# Import necessary libraries
import torch  # PyTorch library for deep learning and tensor computation
import matplotlib.pyplot as plt  # For plotting images and graphs
import numpy as np  # For numerical operations, specifically array manipulation
from PIL import Image  # Python Imaging Library for opening and handling images
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation  # Transformers library for handling the CLIPSeg model

# Define semantic segmentation labels with their IDs and corresponding colors.
# Each label represents an object class and is assigned a unique color for visualization.
labels = [
    {'name': 'road', 'id': 0, 'color': (128, 64, 128)},
    {'name': 'sidewalk', 'id': 1, 'color': (244, 35, 232)},
    {'name': 'building', 'id': 2, 'color': (70, 70, 70)},
    {'name': 'wall', 'id': 3, 'color': (102, 102, 156)},
    {'name': 'fence', 'id': 4, 'color': (190, 153, 153)},
    {'name': 'pole', 'id': 5, 'color': (153, 153, 153)},
    {'name': 'traffic light', 'id': 6, 'color': (250, 170, 30)},
    {'name': 'traffic sign', 'id': 7, 'color': (220, 220, 0)},
    {'name': 'vegetation', 'id': 8, 'color': (107, 142, 35)},
    {'name': 'terrain', 'id': 9, 'color': (152, 251, 152)},
    {'name': 'sky', 'id': 10, 'color': (70, 130, 180)},
    {'name': 'person', 'id': 11, 'color': (220, 20, 60)},
    {'name': 'rider', 'id': 12, 'color': (255, 0, 0)},
    {'name': 'car', 'id': 13, 'color': (0, 0, 142)},
    {'name': 'truck', 'id': 14, 'color': (0, 0, 70)},
    {'name': 'bus', 'id': 15, 'color': (0, 60, 100)},
    {'name': 'train', 'id': 16, 'color': (0, 80, 100)},
    {'name': 'motorcycle', 'id': 17, 'color': (0, 0, 230)},
    {'name': 'bicycle', 'id': 18, 'color': (119, 11, 32)},
    {'name': 'void', 'id': 29, 'color': (0, 0, 0)},  # Used for undefined or ignored areas
]

# Create dictionaries for easy lookup of label IDs to colors and names.
label_colors = {label['id']: label['color'] for label in labels}
label_names = {label['id']: label['name'] for label in labels}

# Load the pre-trained CLIPSeg processor and model from Hugging Face's model hub.
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Specify the path to the local image to be processed.
image_path = r"C:\Users\bassogio\OneDrive\שולחן העבודה\university\robert\prepared_data\train\images\aachen_aachen_000000_000019_leftImg8bit.png"

# Open the image using PIL and convert it to a format suitable for processing.
image = Image.open(image_path)

# Extract all label names from the 'labels' list for use in the model's input prompts.
prompts = [label['name'] for label in labels]  # List of all label names for text prompts

# Prepare the inputs for the CLIPSeg model by processing the image and the label prompts.
# The image is replicated for each label to create input pairs for inference.
inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

# Run the model in inference mode (no gradient computation).
with torch.no_grad():
    outputs = model(**inputs)

# Extract the logits from the model's output, which represents the raw predictions.
preds = outputs.logits  # Shape: [batch_size, num_labels, height, width]

# Choose one of the following for processing 'preds'
# 1. Sigmoid Activation
processed_preds = torch.sigmoid(preds)
# 2. Softmax Activation
#processed_preds = torch.softmax(preds, dim=1)
# 3. ReLU Activation
#processed_preds = torch.relu(preds)
# 4. Log-Softmax Activation
#processed_preds = torch.nn.functional.log_softmax(preds, dim=1)
# 5. Thresholding for binary masks
#threshold = 0.5
#processed_preds = (torch.sigmoid(preds) > threshold).int()
# 6. Normalization (Min-Max Scaling)
# Gave results similar to sigmoid
#processed_preds = (preds - preds.min()) / (preds.max() - preds.min())
# 7. Softplus Activation
# Gave results similar to sigmoid
#processed_preds = torch.nn.functional.softplus(preds)
# 8. Max pooling operation 
#processed_preds = torch.nn.functional.max_pool2d(preds, kernel_size=2, stride=2)
# 9. Leaky ReLU
# Gave results similar to sigmoid
#processed_preds = torch.nn.functional.leaky_relu(preds, negative_slope=0.01)

# Determine the predicted class for each pixel by finding the class with the highest probability.
# The `argmax` operation finds the index of the maximum value along the class dimension (second dimension).
combined_preds = processed_preds.squeeze(0).argmax(dim=0) 

# Create an empty image array to store the colored mask for visualization.
colored_mask = np.zeros((combined_preds.shape[0], combined_preds.shape[1], 3), dtype=np.uint8)

# Map each pixel in the prediction to its corresponding color based on the label ID.
for label_id, color in label_colors.items():
    colored_mask[combined_preds == label_id] = color

# Function to display label information when the mouse moves over the image.
# It shows the label of the pixel the mouse is hovering over in the second plot.
def on_mouse_move(event):
    if event.inaxes == ax2:  # Check if the mouse is within the second plot (segmentation mask)
        x, y = int(event.xdata), int(event.ydata)
        # Ensure the coordinates are within the bounds of the image.
        if 0 <= x < combined_preds.shape[1] and 0 <= y < combined_preds.shape[0]:
            label_id = combined_preds[y, x].item()  # Get the label ID at the current pixel
            label_name = label_names[label_id]  # Get the name of the label
            # Set the title of the plot to show the label name.
            ax2.set_title(f"Label: {label_name}", fontsize=10, color='black')
            fig.canvas.draw()  # Redraw the canvas to update the title

# Create a figure with two subplots to display the original image and the segmentation mask side by side.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Plot the original image on the first subplot.
ax1.imshow(image)
ax1.axis('off')  # Hide the axes for a cleaner look.
ax1.set_title("Original Image")

# Plot the segmentation mask on the second subplot.
ax2.imshow(colored_mask)
ax2.axis('off')  # Hide the axes for a cleaner look.
ax2.set_title("Segmentation Mask")

# Connect the mouse movement event to the function for displaying label info.
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# Display the figure with the two subplots.
plt.show()
