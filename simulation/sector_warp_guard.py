"""Device-only sticky state checks for the experimental sector model."""
import warp as wp

@wp.kernel
def scan_state(q:wp.array2d(dtype=wp.float32),v:wp.array2d(dtype=wp.float32),flags:wp.array(dtype=wp.int32),nq:int,nv:int,h0:int,h1:int,shaft:int,initial0:float,initial1:float,initial_shaft:float,hip_limit:float,gear_limit:float):
 w=wp.tid();bits=flags[w]
 for i in range(nq):
  if not wp.isfinite(q[w,i]):bits=bits|1
 for i in range(nv):
  if not wp.isfinite(v[w,i]):bits=bits|1
 d0=q[w,h0]-initial0;d1=q[w,h1]-initial1
 if wp.abs(d0)>hip_limit or wp.abs(d1)>hip_limit:bits=bits|2
 if wp.abs(d0+.5*(q[w,shaft]-initial_shaft))>gear_limit:bits=bits|4
 flags[w]=bits


@wp.kernel
def scan_contacts(geom:wp.array(dtype=wp.vec2i),worldid:wp.array(dtype=wp.int32),force:wp.array(dtype=wp.spatial_vector),flags:wp.array(dtype=wp.int32),count:int,nworld:int,ground:int,limit:float):
 i=wp.tid()
 if i<count:
  w=worldid[i]
  if w>=0 and w<nworld and geom[i][0]!=ground and geom[i][1]!=ground and force[i][0]>limit:
   wp.atomic_or(flags,w,8)

@wp.kernel
def restore_masked_state(q:wp.array2d(dtype=wp.float32),v:wp.array2d(dtype=wp.float32),ctrl:wp.array2d(dtype=wp.float32),targets:wp.array2d(dtype=wp.float32),integral:wp.array2d(dtype=wp.float32),flags:wp.array(dtype=wp.int32),mask:wp.array(dtype=wp.bool),initial_q:wp.array(dtype=wp.float32),initial_target:wp.array(dtype=wp.float32),nq:int,nv:int,nu:int):
 w=wp.tid()
 if mask[w]:
  for i in range(nq):q[w,i]=initial_q[i]
  for i in range(nv):v[w,i]=0.
  for i in range(nu):
   ctrl[w,i]=0.
   targets[w,i]=initial_target[i]
   integral[w,i]=0.
  flags[w]=0

@wp.kernel
def set_masked_qpos(q:wp.array2d(dtype=wp.float32),poses:wp.array2d(dtype=wp.float32),mask:wp.array(dtype=wp.bool),nq:int):
 w=wp.tid()
 if mask[w]:
  for i in range(nq):q[w,i]=poses[w,i]
