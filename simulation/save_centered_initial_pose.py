from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np,mujoco
from PIL import Image
p=Path(r'C:\Practice_project\biped_sim');t=E.parse(p/'biped_chassis_collision_candidate.xml');root=t.getroot();root.set('model','biped_centered_initial_pose');q=np.load(p/'balanced_initial_pose_exact.npz')['qpos'];keys=E.SubElement(root,'keyframe');E.SubElement(keys,'key',name='centered_start',qpos=' '.join(f'{v:.16g}' for v in q));E.indent(root);out=p/'biped_centered_initial_pose.xml';t.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);print('valid key',m.nkey,'initial contacts',d.ncon,'max initial penetration mm',1000*min([d.contact[i].dist for i in range(d.ncon)]+[0]),'finite',np.isfinite(d.qpos).all())
r=mujoco.Renderer(m,480,640);cam=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(m,cam);cam.lookat[:]=[.04,-.02,.17];cam.distance=.7;cam.azimuth=90;cam.elevation=-7;r.update_scene(d,camera=cam);Image.fromarray(r.render()).save(p/'centered_pose_preview.png')
