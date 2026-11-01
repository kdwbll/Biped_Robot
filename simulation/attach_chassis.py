from pathlib import Path
import xml.etree.ElementTree as E
import mujoco
from PIL import Image
p=Path(__file__).parent; f=next((p/'meshes_original_coords').glob('*2026-04-23-16-40-03-958-1.STL'))
tree=E.parse(p/'two_legs_fixed_base_experiment.xml'); root=tree.getroot(); E.SubElement(root.find('asset'),'mesh',name='chassis_mesh',file=str(f.relative_to(p)).replace('\\','/'),scale='0.001 0.001 0.001'); E.SubElement(root.find('worldbody'),'geom',name='chassis_visual',type='mesh',mesh='chassis_mesh',contype='0',conaffinity='0',rgba='0.6 0.65 0.72 1')
E.indent(root); path=p/'two_legs_fixed_base_experiment.xml';tree.write(path,encoding='utf8',xml_declaration=True);m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
r=mujoco.Renderer(m,480,640);r.update_scene(d);Image.fromarray(r.render()).save(p/'two_legs_preview.png');print('meshes',m.nmesh,'rendered')
