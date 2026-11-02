"""Experimental GPU PI servo kernels; see parity checks before use."""
import warp as wp

@wp.kernel
def update_targets(initial:wp.array2d(dtype=wp.float32),targets:wp.array2d(dtype=wp.float32),delta_us:float):
 w,j=wp.tid();change=wp.float32(0.)
 if j==4:change=wp.float32(delta_us/2000.)*wp.float32(4.71238898038469)
 desired=initial[w,j]+change;limit=wp.float32(1.0471975511965976/.26*.02)
 targets[w,j]+=wp.clamp(desired-targets[w,j],-limit,limit)

@wp.kernel
def servo_step(targets:wp.array2d(dtype=wp.float32),integral:wp.array2d(dtype=wp.float32),length:wp.array2d(dtype=wp.float32),velocity:wp.array2d(dtype=wp.float32),ctrl:wp.array2d(dtype=wp.float32),dt:float,hip_kd:float):
 w,j=wp.tid();kp=wp.float32(4.);ki=wp.float32(1.5);kd=wp.float32(.08)
 if j==4:kp=wp.float32(8.);ki=wp.float32(6.);kd=wp.float32(hip_kd)
 error=targets[w,j]-length[w,j];vel=velocity[w,j];raw=kp*error-kd*vel+integral[w,j];speed=wp.float32(1.0471975511965976/.26);stall=wp.float32(24.*.0980665);available=stall
 if raw*vel>0.:available=stall*wp.max(0.,1.-wp.abs(vel)/speed)
 torque=wp.clamp(raw,-available,available);permit=wp.abs(raw)<=available or (raw-torque)*error<0.
 if permit:integral[w,j]=wp.clamp(integral[w,j]+ki*error*wp.float32(dt),-1.,1.)
 ctrl[w,j]=torque


@wp.kernel
def update_pwm_targets(initial:wp.array2d(dtype=wp.float32),targets:wp.array2d(dtype=wp.float32),pwm:wp.array2d(dtype=wp.float32)):
 w,j=wp.tid()
 desired=initial[w,j]+(pwm[w,j]-1500.)/2000.*wp.float32(4.71238898038469)
 limit=wp.float32(1.0471975511965976/.26*.02)
 targets[w,j]+=wp.clamp(desired-targets[w,j],-limit,limit)
