"""Create a simulated tilted initial pose with no initial floor penetration."""
import mujoco
import numpy as np

def tilted_pose(model, original_qpos, ground, roll_deg, pitch_deg):
    q = np.asarray(original_qpos, dtype=float).copy()
    # World rotations: roll then pitch, applied to the initial free-joint quaternion.
    def multiply(a, b):
        w,x,y,z=a; v,i,j,k=b
        return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                         w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])
    roll = np.deg2rad(roll_deg)/2
    pitch = np.deg2rad(pitch_deg)/2
    q[3:7] = multiply([np.cos(pitch),0,np.sin(pitch),0],
                       multiply([np.cos(roll),np.sin(roll),0,0],q[3:7]))
    probe = mujoco.MjData(model)
    probe.qpos[:] = q
    mujoco.mj_forward(model, probe)
    distances = [c.dist for c in probe.contact[:probe.ncon] if ground in (c.geom1,c.geom2)]
    lift = max(0., -min(distances, default=0.))
    q[2] += lift
    return q.astype(np.float32), float(lift)
