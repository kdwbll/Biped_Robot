"""Sample ground-contact point slip, using MuJoCo world-frame spatial velocities."""
import numpy as np

def sole_sample_slip(geom, worlds, positions, normal_force, cvel, subtree_com,
                     ground, soles, bodies, roots, nworld, threshold_n=.5):
    """Return per-world, per-foot load and load-weighted planar contact speed.

    The ground is fixed and horizontal. cvel is referenced to the kinematic root's
    subtree COM, not to the body origin. Threshold applies to summed normal load.
    """
    geom=np.asarray(geom);worlds=np.asarray(worlds,dtype=int);positions=np.asarray(positions)
    normal=np.maximum(0.,np.asarray(normal_force));load=np.zeros((nworld,2));weighted=np.zeros((nworld,2))
    for leg,sole in enumerate(soles):
        mask=(((geom[:,0]==ground)&(geom[:,1]==sole))|((geom[:,1]==ground)&(geom[:,0]==sole)))
        mask &= (worlds>=0)&(worlds<nworld)&(normal>0)
        if not np.any(mask):continue
        w=worlds[mask];spatial=cvel[w,bodies[leg]];com=subtree_com[w,roots[leg]]
        velocity=spatial[:,3:]+np.cross(spatial[:,:3],positions[mask]-com)
        np.add.at(load[:,leg],w,normal[mask])
        np.add.at(weighted[:,leg],w,normal[mask]*np.linalg.norm(velocity[:,:2],axis=1))
    speed=np.divide(weighted,load,out=np.zeros_like(weighted),where=load>=threshold_n)
    return load,speed
