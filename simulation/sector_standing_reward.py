"""Experimental standing-only reward shared by CPU and GPU environments."""
import numpy as np

def shaped_standing_reward(upright, height, initial_height, angular_velocity, action, previous_action, terminated):
    """Score uprightness, height, angular motion and smooth actions at a 20 ms control interval.

    Numerical coefficients are hypotheses for simulation experiments, not physical limits.
    This reward contains no forward-walking objective.
    """
    upright = np.asarray(upright, dtype=float)
    drop = np.maximum(0., np.asarray(initial_height, dtype=float)-np.asarray(height, dtype=float))
    omega = np.asarray(angular_velocity, dtype=float)
    action = np.asarray(action, dtype=float)
    previous = np.asarray(previous_action, dtype=float)
    score = (np.maximum(0.,upright) - .5*(drop/.08)**2
             - .025*np.mean(omega**2,axis=-1)
             - .01*np.mean(action**2,axis=-1)
             - .02*np.mean((action-previous)**2,axis=-1))
    return .02*score-np.asarray(terminated,dtype=float)
