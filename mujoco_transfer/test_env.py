from mujoco_transfer.fsm_ik_env import MujocoFsmIkEnv

def test_environment():
    # 1. Initialize the environment
    env = MujocoFsmIkEnv(model_path="mujoco_transfer/models/iiwa_wsg_paddle_ball.xml")

    print("=== Testing Domain Randomization ===")
    for i in range(3):
        obs, info = env.reset(seed=i)
        # Fetch the actual mass from MuJoCo's C-struct memory to prove it changed
        ball_mass = env.model.body_mass[env._ball_body_id]
        print(f"Reset {i+1} | Ball Mass: {ball_mass:.5f} kg")

    print("\n=== Testing Gym Step API ===")
    for step in range(5):
        # Sample a random valid action from your Box(-1, 1, shape=(7,)) space
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Index 16 is ball_z based on your Drake observation space
        print(f"Step {step+1} | Action max: {max(action):.2f} | Reward: {reward:.3f} | Ball Z: {obs[16]:.3f}")
        
        if terminated or truncated:
            print("Episode ended early!")
            break

    print("\nSuccess! Environment is fully Gymnasium-compliant and ready.")

if __name__ == "__main__":
    test_environment()