"""Experimental batched version of theoretical_servo.py. Simulation only."""
import math
import numpy as np
import warp as wp
import mujoco_warp as mjw

TRAVEL=float(math.radians(270.))
SPEED=float(math.radians(60.)/.26)
STALL=float(24.*.0980665)
LIMIT=SPEED*.02

@wp.kernel
def update_targets(initial:wp.array2d(dtype=wp.float32),targets:wp.array2d(dtype=wp.float32),actions:wp.array2d(dtype=wp.float32)):
 world,j=wp.tid()
 desired=initial[world,j]+wp.clamp(actions[world,j],-1.0,1.0)*(180.0/2000.0)*TRAVEL
 targets[world,j]=targets[world,j]+wp.clamp(desired-targets[world,j],-LIMIT,LIMIT)

@wp.kernel
def servo_control(targets:wp.array2d(dtype=wp.float32),length:wp.array2d(dtype=wp.float32),
                  velocity:wp.array2d(dtype=wp.float32),ctrl:wp.array2d(dtype=wp.float32)):
 world,j=wp.tid()
 vel=velocity[world,j];kd=wp.float32(.08)
 if j==4:kd=wp.float32(.01)
 raw=(targets[world,j]-length[world,j])-kd*vel
 available=STALL
 if raw*vel>0.0:
  available=STALL*wp.max(0.0,1.0-wp.abs(vel)/SPEED)
 ctrl[world,j]=wp.clamp(raw,-available,available)

@wp.kernel
def support_force(xmat:wp.array2d(dtype=wp.mat33),cvel:wp.array2d(dtype=wp.spatial_vector),
                  xfrc:wp.array2d(dtype=wp.spatial_vector),body:int,gain:float,weight_force:float):
 world=wp.tid();rot=xmat[world,body];vel=cvel[world,body]
 tx=wp.clamp(gain*(rot[1,2]-.18*vel[0]),-1.5,1.5)
 ty=wp.clamp(gain*(-rot[0,2]-.18*vel[1]),-1.5,1.5)
 xfrc[world,body]=wp.spatial_vector(0.,0.,weight_force,tx,ty,0.)

class WarpServo:
 def __init__(self,model,data,cpu_initial,nworld):
  self.model=model;self.data=data;self.nworld=nworld
  initial=np.tile(np.asarray(cpu_initial,dtype=np.float32),(nworld,1))
  self.initial=wp.array(initial,dtype=wp.float32,device='cuda:0')
  self.targets=wp.array(initial,dtype=wp.float32,device='cuda:0')
  self.actions=wp.zeros((nworld,5),dtype=wp.float32,device='cuda:0')
  # One graph advances one physics step with the same feedback law as CPU.
  with wp.ScopedCapture() as capture:
   wp.launch(servo_control,dim=(nworld,5),inputs=[self.targets,data.actuator_length,data.actuator_velocity,data.ctrl])
   mjw.step(model,data)
  self.graph=capture.graph
 def set_actions(self,actions):
  a=np.asarray(actions,dtype=np.float32)
  if a.shape!=(self.nworld,5) or not np.isfinite(a).all():raise ValueError('Expected finite nworld x 5 actions')
  wp.copy(self.actions,wp.array(a,dtype=wp.float32,device='cuda:0'))
  wp.launch(update_targets,dim=(self.nworld,5),inputs=[self.initial,self.targets,self.actions])
 def set_actions_device(self,actions):
  # Caller synchronizes the originating CUDA stream before this zero-copy view.
  wp.copy(self.actions,wp.from_torch(actions.contiguous(),dtype=wp.float32))
  wp.launch(update_targets,dim=(self.nworld,5),inputs=[self.initial,self.targets,self.actions])
 def control_step(self):
  for _ in range(40):wp.capture_launch(self.graph)
 def apply_support(self,body,gain,weight_force):
  wp.launch(support_force,dim=self.nworld,inputs=[self.data.xmat,self.data.cvel,self.data.xfrc_applied,body,gain,weight_force])
