# Optimization-Image-Segmentation-Using-Linear-Programming
We developed an image segmentation tool using linear programming and the Max Flow/Min Cut theorem to separate the foreground from the background in grayscale images.

The goal was to formulate the image segmentation problem as a flow network and solve it using optimization techniques, providing a robust solution to segment images efficiently.

---

### Problem Description:
We treated the image as a **2D grid of pixel intensities**, where each pixel is modeled as a node, and the similarity between neighboring pixels is represented as edges. The goal was to use **Max Flow/Min Cut** theory to find the optimal cuts that would separate the foreground from the background.

### Methodology:

1. **Network Creation**:
   - Each pixel was represented as a **node** in a network, and connections were established between neighboring pixels based on their intensity similarity.
   - The similarity between two neighboring pixels was calculated using an **exponential function** of their intensity difference.

2. **Formulation of the Linear Program**:
   - We formulated the **linear programming (LP)** problem by introducing decision variables for each non-zero link in the pixel network.
   - The objective was to maximize the flow from the **source node** (background) to the **sink node** (foreground), constrained by flow conservation at each pixel.
   - The **Gurobi optimizer** was used to solve the max flow problem.

3. **Residual Network & Cut Identification**:
   - After solving the LP, we calculated the **residual network** to identify minimal cuts separating the foreground and background.
   - A **depth-first search** was employed to identify nodes accessible from the source, with the cuts representing optimal separation.

4. **Segmentation Results**:
   - The tool segmented images into foreground and background regions, visualizing cuts by marking them in red lines to clearly distinguish between the two regions.

5. **Testing & Evaluation**:
   - The tool was tested on various images, performing effectively in identifying boundaries between foreground and background.
   - Compared to **Photoshop’s Magic Wand**, our solution provided more precise cuts and efficiency for users without advanced photo editing skills.

---

### Key Findings:
- **Efficiency**: The LP-based approach efficiently handled both simple and complex images by leveraging sparse matrices to reduce computational complexity.
- **Segmentation Accuracy**: The segmentation results were robust, offering clear and accurate cuts.
- **Scalability**: The tool scaled well and proved suitable for a wide range of image segmentation tasks.

---

### Future Improvements:
- **Parallel Processing**: Implementing GPU-based acceleration for handling larger, high-resolution images.
- **Complex Image Segmentation**: Fine-tuning the model to improve accuracy for more complex images with occlusions.

---

### Conclusion:
This project demonstrated the successful application of **linear programming** and the **Max Flow/Min Cut theorem** for **image segmentation**. The tool provided an effective and accessible solution for separating foreground and background, with potential for future scalability and improvements.

---

### Group Members:
- **Sushanth Ravichandran**
- **Oliver Gault**
- **Gayathree Gopi**
- **Brooks Li**

