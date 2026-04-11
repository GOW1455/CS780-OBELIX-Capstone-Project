import cv2
import torch
import numpy as np
import pickle
from obelix import OBELIX # Assuming your obelix.py is in the same folder

# Map your keyboard to the environment actions
# w = Forward, a = Left 45, d = Right 45, q = Left 22, e = Right 22
KEY_MAPPING = {
    ord('a'): 0, # L45
    ord('q'): 1, # L22
    ord('w'): 2, # FW
    ord('e'): 3, # R22
    ord('d'): 4  # R45
}

def main():
    env = OBELIX(scaling_factor=5, difficulty=0, wall_obstacles=False)
    
    recorded_episodes = []
    
    print("--- HUMAN DATA COLLECTION ---")
    print("Controls: W=Forward, A=Left45, D=Right45, Q=Left22, E=Right22")
    print("Press 'ESC' to save and quit.")
    
    for ep in range(10): # Record 10 successful episodes
        state = env.reset()
        env.render_frame() # Force initial render
        
        ep_states = []
        ep_actions = []
        done = False
        
        while not done:
            # Wait indefinitely for a valid keypress
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27: # ESC key
                print("Exiting and saving...")
                with open("human_data.pkl", "wb") as f:
                    pickle.dump(recorded_episodes, f)
                return
                
            if key in KEY_MAPPING:
                action_idx = KEY_MAPPING[key]
                
                # Record what the human saw, and what the human did
                ep_states.append(state)
                ep_actions.append(action_idx)
                
                # Step the environment
                state, reward, done_info = env.step(["L45", "L22", "FW", "R22", "R45"][action_idx], render=True)
                done = bool(done_info)
                
        print(f"Episode {ep+1} Finished. Recorded {len(ep_states)} steps.")
        recorded_episodes.append((ep_states, ep_actions))

    # Save data
    with open("human_data.pkl", "wb") as f:
        pickle.dump(recorded_episodes, f)
    print("Saved 10 episodes to human_data.pkl")

if __name__ == "__main__":
    main()