import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Function to calculate similarity between two pixel intensities
def similarity(intensity_i, intensity_j, sigma):
    return np.ceil(100 * np.exp(-((intensity_i - intensity_j) ** 2) / (2 * sigma ** 2)))

# Create network matrix
def create_network(image, sigma):
    rows, cols = image.shape
    num_pixels = rows * cols
    network = np.zeros((num_pixels, num_pixels))

    for r in range(rows):
        for c in range(cols):
            current_pixel = r * cols + c
            if r > 0:  # Up
                network[current_pixel, (r - 1) * cols + c] = similarity(image[r, c], image[r - 1, c], sigma)
            if r < rows - 1:  # Down
                network[current_pixel, (r + 1) * cols + c] = similarity(image[r, c], image[r + 1, c], sigma)
            if c > 0:  # Left
                network[current_pixel, r * cols + c - 1] = similarity(image[r, c], image[r, c - 1], sigma)
            if c < cols - 1:  # Right
                network[current_pixel, r * cols + c + 1] = similarity(image[r, c], image[r, c + 1], sigma)

    return network

# Function to add source and sink nodes to the network
def add_source_sink(network, background_pixel, foreground_pixel):
    num_pixels = network.shape[0]
    max_similarity = np.max(network)
    new_network = np.zeros((num_pixels + 2, num_pixels + 2))

    # Copy original network into the new one
    new_network[:num_pixels, :num_pixels] = network

    # Add source and sink nodes
    new_network[-2, background_pixel] = max_similarity
    new_network[foreground_pixel, -1] = max_similarity

    return new_network

# Function to solve the max flow problem using Gurobi
def solve_max_flow(network):
    num_nodes = network.shape[0]

    # Initialize the Gurobi model
    model = gp.Model("max_flow")

    # Decision variables: flow through each link in the network
    flow = model.addVars(num_nodes, num_nodes, ub=network, name="flow", vtype=GRB.CONTINUOUS)

    # Objective: maximize the flow out of the source node (-2)
    model.setObjective(gp.quicksum(flow[num_nodes-2, j] for j in range(num_nodes)), GRB.MAXIMIZE)

    # Constraints: Flow conservation for each node (except source and sink)
    for i in range(num_nodes - 2):  # Skip source and sink
        model.addConstr(gp.quicksum(flow[i, j] for j in range(num_nodes)) == gp.quicksum(flow[j, i] for j in range(num_nodes)))

    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Max Flow: {model.objVal}")
        return model, flow
    else:
        print("Optimization did not find a solution.")
        return None, None

# Function to find background and foreground pixels
def find_background_foreground(image):
    # The background is the pixel with the minimum intensity (assumed to be the darkest pixel)
    # The foreground is the pixel with the maximum intensity (assumed to be the brightest pixel)
    background_pixel = np.argmin(image)  # Flatten the image and find the index of the minimum intensity
    foreground_pixel = np.argmax(image)  # Flatten the image and find the index of the maximum intensity
    return background_pixel, foreground_pixel

# Load image from CSV
def load_image_from_csv(filepath):
    # Assuming the CSV file contains the pixel intensity matrix
    image = pd.read_csv(filepath, header=None).to_numpy()
    return image

# Function to calculate residual network
def calculate_residual_network(network, flow, num_pixels):
    residual_network = np.zeros_like(network)
    for i in range(num_pixels):
        for j in range(num_pixels):
            residual_network[i, j] = network[i, j] - flow[i, j].X  # Subtract actual flow from max capacity
    network       
    return residual_network

# Depth-first search to find accessible nodes in residual network
def depth_first_search(residual_network, start_node):
    visited = set()
    stack = [start_node]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            neighbors = np.nonzero(residual_network[node])[0]
            stack.extend(neighbors)
    
    return visited

# Main function to run image segmentation
def image_segmentation(image, background_pixel, foreground_pixel, sigma=0.05):
    network = create_network(image, sigma)
    network_with_source_sink = add_source_sink(network, background_pixel, foreground_pixel)
    result, flow = solve_max_flow(network_with_source_sink)
    
    if result:
        num_pixels = image.size
        residual_network = calculate_residual_network(network_with_source_sink, flow, num_pixels + 2)
        
        # Perform DFS from source node (-2 in the array)
        source_accessible = depth_first_search(residual_network, num_pixels)  # DFS from the source node
        
        return source_accessible, residual_network, flow, network_with_source_sink

    return None, None, None, None

# Function to plot image and cuts
def plot_image_with_cuts(image, network, source_accessible):
    rows, cols = image.shape
    cut_edges = np.zeros_like(image)

    for r in range(rows):
        for c in range(cols):
            current_pixel = r * cols + c
            if current_pixel in source_accessible:
                # Find neighboring pixels not accessible from source
                if r > 0 and (r - 1) * cols + c not in source_accessible:
                    cut_edges[r, c] = 1  # Mark cut
                if r < rows - 1 and (r + 1) * cols + c not in source_accessible:
                    cut_edges[r, c] = 1  # Mark cut
                if c > 0 and r * cols + c - 1 not in source_accessible:
                    cut_edges[r, c] = 1  # Mark cut
                if c < cols - 1 and r * cols + c + 1 not in source_accessible:
                    cut_edges[r, c] = 1  # Mark cut

    # Plot the original image
    # plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray', interpolation='nearest')

    # Overlay the cuts as a contour line in red
    plt.contour(cut_edges, levels=[0.5], colors='red', linewidths=2)  # Thin red contour line
    plt.title('Image with Cut Contour')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load the image from a CSV file
    image = load_image_from_csv("oval-1.csv")  # Replace with the actual file path

    # Find the background and foreground pixels automatically
    background_pixel, foreground_pixel = find_background_foreground(image)

    # Run the image segmentation
    source_accessible, residual_network, flow, network_with_source_sink = image_segmentation(image, background_pixel, foreground_pixel)

    if source_accessible:
        # Plot the image with cuts
        plot_image_with_cuts(image, network_with_source_sink, source_accessible)