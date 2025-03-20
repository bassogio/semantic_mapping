## Rotation Matrix Overview - Used in point_cloud_config

This project involves working with rotation matrices that transform 3D coordinates from one coordinate system to another. A rotation matrix is a tool that describes how a point in 3D space moves after a rotation around an axis. 

A rotation matrix is typically a 3x3 matrix, where each element defines how much one of the original axes (x, y, or z) contributes to the new axes after rotation.

### General Form of a Rotation Matrix

The rotation matrix can be represented as follows:

| a  b  c |  
| d  e  f |  
| g  h  i |  

### Elements of the Rotation Matrix

1. **a**:
   - Controls how much the original x-axis influences the new x-axis.
   - **Possible values**: -1, 0, 1
   - **Interpretation**:
     - `a = 1`: The x-axis remains unchanged.
     - `a = 0`: The x-axis does not contribute to the new x-axis.
     - `a = -1`: The x-axis is flipped to the opposite direction (e.g., 180° rotation around the Y-axis).

2. **b**:
   - Determines how much the original y-axis affects the new x-axis.
   - **Possible values**: -1, 0, 1
   - **Interpretation**:
     - `b > 0`: Positive values mean that the y-axis "tilts" the x-axis in the positive x-direction.
     - `b < 0`: Negative values cause the y-axis to tilt the x-axis in the negative x-direction.

3. **c**:
   - Determines how much the original z-axis contributes to the new x-axis.
   - **Possible values**: -1, 0, 1
   - **Interpretation**:
     - `c > 0`: Positive values cause the z-axis to influence the x-axis in a positive direction.
     - `c < 0`: Negative values cause the z-axis to influence the x-axis in the opposite direction.

4. **d**:
   - Determines how much the original x-axis contributes to the new y-axis.
   - **Possible values**: -1, 0, 1
   - **Interpretation**:
     - `d > 0`: Positive values mean the x-axis "tilts" the y-axis in the positive y-direction.
     - `d < 0`: Negative values cause the x-axis to tilt the y-axis in the negative y-direction.

5. **e**:
   - Controls how much the original y-axis contributes to the new y-axis.
   - **Possible values**: -1, 0, 1
   - **Interpretation**:
     - `e = 1`: The y-axis remains unchanged.
     - `e = 0`: The y-axis does not influence the new y-axis.
     - `e = -1`: The y-axis is flipped.

6. **f**:
   - Determines how much the original z-axis contributes to the new y-axis.
   - **Possible values**: -1, 0, 1
   - **Interpretation**:
     - `f > 0`: Positive values cause the z-axis to influence the y-axis in the positive direction.
     - `f < 0`: Negative values cause the z-axis to influence the y-axis in the opposite direction.

7. **g**:
   - Determines how much the original x-axis contributes to the new z-axis.
   - **Possible values**: -1, 0, 1
   - **Interpretation**:
     - `g > 0`: Positive values mean the x-axis influences the z-axis in the positive direction.
     - `g < 0`: Negative values cause the x-axis to influence the z-axis in the opposite direction.

8. **h**:
   - Determines how much the original y-axis contributes to the new z-axis.
   - **Possible values**: -1, 0, 1
   - **Interpretation**:
     - `h > 0`: Positive values cause the y-axis to influence the z-axis in the positive direction.
     - `h < 0`: Negative values cause the y-axis to influence the z-axis in the opposite direction.

9. **i**:
   - Controls how much of the original z-axis remains in the new z-axis.
   - **Possible values**: -1, 0, 1
   - **Interpretation**:
     - `i = 1`: The z-axis remains unchanged.
     - `i = 0`: The z-axis does not contribute to the new z-axis.
     - `i = -1`: The z-axis is flipped.

