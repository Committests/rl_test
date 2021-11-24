import gym

from wrappers import (
    NormalizeWrapper,
    ImgWrapper,
    DtRewardWrapper,
    ActionWrapper,
    ResizeWrapper,
    SteeringToWheelVelWrapper,
    ObsWrapper,
   
)


def launch_env(id=None):
    env = None
    if id is None:
        from gym_duckietown.simulator import Simulator

        env = Simulator(
            seed=123,  # random seed
            map_name="zigzag_dists",#loop_empty Duckietown-zigzag_dists-v0 ETHZ_autolab_technical_track, zigzag_dists
            max_steps=500001,  # we don't want the gym to reset itself
            domain_rand=True, #-------------------was false
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            distortion=True,
            randomize_maps_on_reset=False,#false
            draw_curve=False,#false
            draw_bbox=False,#false
            frame_skip=4,#goes faster
            draw_DDPG_features=False,##--------------added
        )
    else:
        env = gym.make(id)
    
    # Wrappers
    env= ObsWrapper(env)
    env = ResizeWrapper(env)
    #env = NormalizeWrapper(env) #remooved
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
 
    env = SteeringToWheelVelWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)

    return env
