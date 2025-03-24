import numpy as np

def select_goal_and_update_prob(goal_dist_prob, decay_rate=0.25):
    # Flatten the goal_dist_prob to easily pick one
    flat_prob = goal_dist_prob.flatten()
    
    # Randomly choose a goal index based on the current probability distribution
    chosen_goal_idx = np.random.choice(len(flat_prob), p=flat_prob)
    
    # Convert the flat index back to 2D coordinates (row, column)
    row, col = divmod(chosen_goal_idx, goal_dist_prob.shape[1])
    
    # Decay the probability of the chosen goal by a certain factor
    goal_dist_prob[row, col] *= (1 - decay_rate)
    
    # Normalize the probabilities so they sum to 1
    goal_dist_prob /= goal_dist_prob.sum()
    
    return (row, col), goal_dist_prob

# Example goal_dist_prob matrix
goal_dist_prob = np.array([[0.1, 0.2, 0.3],
                           [0.15, 0.05, 0.2]])

goals = np.array([[[10, 40],[20, 40],[30, 40],[40, 40]],
                  [[10, 30],[20, 30],[30, 30],[40, 30]],
                  [[10, 20],[20, 20],[30, 20],[40, 20]],
                  [[10, 10],[20, 10],[30, 10],[40, 10]]])

# Call the function and assign the output to `goal` and `goal_dist_prob`
goal, goal_dist_prob = select_goal_and_update_prob(goal_dist_prob)

# `goal` now contains the selected goal as (row, col)
print("Selected goal:", goals[goal])

# Updated probability distribution
print("Updated goal distribution:", goal_dist_prob)
