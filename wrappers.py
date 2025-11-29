from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info


def apply_wrappers(env, skip_frame, resize, frame_stack):
    env = SkipFrame(env, skip=skip_frame) # Num of frames to apply one action to
    env = ResizeObservation(env, shape=resize) # Resize frame from 240x256 to 84x84
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=frame_stack, lz4_compress=True) # May need to change lz4_compress to False if issues arise
    return env