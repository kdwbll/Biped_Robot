from alternating_biped_env import AlternatingBipedEnv
class LargerActionBipedEnv(AlternatingBipedEnv):
    def __init__(self, episode_seconds=4.0, disturbance=.02):
        super().__init__(episode_seconds,disturbance)
        self.action_amplitude_us=180.0
