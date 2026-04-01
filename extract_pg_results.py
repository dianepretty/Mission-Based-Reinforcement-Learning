import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def get_best_reward(log_path):
    try:
        event_acc = EventAccumulator(log_path)
        event_acc.Reload()
        
        # List all available scalar tags in this specific log file
        tags = event_acc.Tags()['scalars']
        
        # Search for the most likely reward tag
        # Usually it's 'rollout/ep_rew_mean', but let's be safe
        reward_tag = None
        for t in tags:
            if 'reward' in t or 'ep_rew_mean' in t:
                reward_tag = t
                break
        
        if reward_tag:
            _, _, vals = zip(*event_acc.Scalars(reward_tag))
            return round(vals[-1], 2), reward_tag
        return None, None
    except Exception as e:
        return None, None

log_base_path = os.path.join("logs", "pg")
results = []

print(f"Scanning {log_base_path}...")

for folder in os.listdir(log_base_path):
    folder_path = os.path.join(log_base_path, folder)
    if os.path.isdir(folder_path):
        reward, tag_used = get_best_reward(folder_path)
        
        # Clean up the name for the CSV
        name_parts = folder.split('_')
        algo = name_parts[0]
        # Reconstruct the experiment name
        exp = "_".join(name_parts[1:-1]) if len(name_parts) > 2 else folder
        
        results.append({
            "Algorithm": algo,
            "Experiment": exp,
            "Reward": reward if reward is not None else "N/A",
            "Tag_Found": tag_used if tag_used else "None",
            "Status": "Complete" if reward is not None else "Failed"
        })

df = pd.DataFrame(results)
df.to_csv("pg_results_final.csv", index=False)
print("Done! If 'Reward' is still N/A, check the 'Tag_Found' column to see what happened.")