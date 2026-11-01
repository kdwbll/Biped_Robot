from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np,mujoco
from PIL import Image
p=Path(r'C:\Practice_project\biped_sim');t=E.parse(p/'biped_chassis_collision_candidate.xml');root=t.getroot();root.set('model','biped_symmetric_pose_approx');q=np.load(p/'balanced_initial_pose.npz')['qpos'].copy();q[2]-=.00260053
keyframe=root.find('keyframe')
if keyframe is None:keyframe=E.SubElement(root,'keyframe')
E.SubElement(keyframe,'key',name='symmetric_start',qpos=' '.join(f'{v:.16g}' for v in q))
E.indent(root);out=p/'biped_symmetric_initial_pose.xml';t.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,x) for x in ('L0_foot_proxy','L1_foot_proxy')];com=d.subtree_com[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')];a,b=d.geom_xpos[ids];print('midpoint xy',(a[:2]+b[:2])/2,'COM xy',com[:2],'difference mm',((a[:2]+b[:2])/2-com[:2])*1000,'height delta mm',(a[2]-b[2])*1000,'initial contacts',d.ncon)
r=mujoco.Renderer(m,480,640);cam=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(m,cam);cam.lookat[:]=[.04,-.02,.17];cam.distance=.7;cam.azimuth=90;cam.elevation=-7;r.update_scene(d,camera=cam);Image.fromarray(r.render()).save(p/'symmetric_initial_pose_preview.png')
