"""Simulation-only guard for the experimentally pruned collision model.

This does not add missing contacts or enforce real motor limits. It stops this
Python-controlled simulation if either hip deviates 4.5 degrees from the
initial keyframe. Direct use of the XML or other controllers bypasses it.
"""
from dataclasses import replace
import numpy as np
import mujoco
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec


class HipRangeExceeded(RuntimeError):
    """Simulation exceeded the guard's conservative hip threshold."""


class GuardedPrunedServo(CandidateStandingServo):
    """PI-D candidate with a fail-stop range check for diagnostic use only."""

    def __init__(self, model, data, limit_deg=4.5):
        if not np.isfinite(limit_deg) or not 0 < limit_deg <= 4.5:
            raise ValueError('Guard threshold must be between 0 and 4.5 degrees')
        required='highres_L0_000' if model.npair==9597 else 'nomerge_L0_000' if model.npair in (10055,11433) else None
        if required is None or mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'nomerge_chassis5_000')<0 or mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,required)<0:
            raise ValueError('This guard only applies to the current pruned diagnostic model')
        if model.npair==11433 and mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'nomerge_chassis0_000')<0:
            raise ValueError('Missing expected chassis0 diagnostic geometry')
        self.hip_qadr=[]
        root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
        if root<0:raise ValueError('Missing robot base')
        for leg in (0,1):
            joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'L{leg}_hip_candidate')
            mount=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'L{leg}_mount')
            part=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'L{leg}_part0')
            if (min(joint,mount,part)<0 or model.body_parentid[mount]!=root or
                model.body_parentid[part]!=mount or model.body_jntnum[part]!=0 or
                model.body_jntnum[mount]!=1 or model.jnt_bodyid[joint]!=mount or
                model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError('Unexpected model hierarchy: hip range assumptions do not apply')
            self.hip_qadr.append(int(model.jnt_qposadr[joint]))
        if model.nkey<1 or not np.allclose(data.qpos[self.hip_qadr],model.key_qpos[0,self.hip_qadr],rtol=0.,atol=1e-9):
            raise ValueError('Initialize both hips from keyframe 0 before enabling the guard')
        self.initial_hips=model.key_qpos[0,self.hip_qadr].copy()
        self.limit_rad=np.deg2rad(limit_deg)
        super().__init__(model,data,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
                         kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
        self.check_range()

    def check_range(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise HipRangeExceeded('Simulation state contains a nonfinite value')
        deviation=self.data.qpos[self.hip_qadr]-self.initial_hips
        if not np.isfinite(deviation).all() or np.any(np.abs(deviation)>self.limit_rad):
            raise HipRangeExceeded(f'Hip range guard stopped simulation: {np.rad2deg(deviation).tolist()} degrees')

    def step(self):
        self.check_range()
        before=float(self.data.time)
        super().step()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or self.data.time<before+self.model.opt.timestep*.5:
            raise HipRangeExceeded('Simulation became nonfinite or was automatically reset')
        self.check_range()
