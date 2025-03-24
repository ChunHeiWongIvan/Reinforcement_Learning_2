import numpy as np

def check_within_distance(coords, distance_threshold=5):
    coords = np.array(coords)
    
    # Iterate over each pair of points
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            # Calculate the Euclidean distance between the pair
            distance = np.linalg.norm(coords[i] - coords[j])
            
            # Check if the distance is less than or equal to the threshold
            if distance <= distance_threshold:
                return True  # Return True if any pair is within the threshold
    
    return False  # No pair found within the threshold

# Example usage:
coordinates = [(1, 2), (3, 4), (5, 6), (1.5, 2.5)]
result = check_within_distance(coordinates, 0.1)
print("Are any sources within 5 meters?", result)
