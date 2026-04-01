import time
import os
from stable_baselines3 import PPO
from environment.custom_env import CivicReportingEnv
from environment.rendering import CivicRenderer

def run_ai_gameplay():
    # 1. Setup the Environment (The "World")
    # We use render_mode="human" to ensure the GUI pops up
    env = CivicReportingEnv(render_mode="human")
    renderer = CivicRenderer(env.institutions)
    
    # 2. Path to your best model
    # Note: SB3 looks for .zip automatically, so we point to the folder + filename
    model_path = os.path.join("models", "best_model", "PPO_Exp4_ShortTerm")
    
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Error: Model not found at {model_path}.zip")
        return

    print(f"🎮 Loading AI Player: {model_path}...")
    model = PPO.load(model_path)

    # 3. Game Loop
    obs, _ = env.reset()
    total_reward = 0
    
    print("🚀 AI is now managing the Rwanda Civic Redress System!")
    
    try:
        while True: # Keep the "game" running
            # AI looks at the current state (obs) and predicts the best action
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Use rendering.py to update the screen
            renderer.render(obs, action, total_reward)
            
            # Slow it down slightly so you can see the decision making (50ms delay)
            time.sleep(0.05) 
            
            if terminated or truncated:
                print(f"🏆 Dispatch Cycle Complete. Total Accountability Score: {total_reward}")
                time.sleep(1) # Pause for a second before restarting
                obs, _ = env.reset()
                total_reward = 0

    except KeyboardInterrupt:
        print("\n🛑 Game Stopped.")
    finally:
        renderer.close()
        env.close()

if __name__ == "__main__":
    run_ai_gameplay()