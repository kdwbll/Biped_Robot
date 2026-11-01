from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
root=Path(__file__).parent
files=sorted((root/'meshes').glob('*.STL'))
m=ET.Element('mujoco',model='biped_static_inspection')
ET.SubElement(m,'compiler',angle='radian')
ET.SubElement(m,'statistic',center='0.1 0.1 0.19',extent='0.5')
a=ET.SubElement(m,'asset'); world=ET.SubElement(m,'worldbody')
ET.SubElement(world,'light',pos='0 0 1',dir='0 0 -1')
ET.SubElement(world,'geom',name='floor',type='plane',size='1 1 0.1',rgba='0.5 0.5 0.5 1')
for i,p in enumerate(files):
 name=f'part{i:02d}'
 ET.SubElement(a,'mesh',name=name,file=str(p.relative_to(root)).replace('\\','/'),scale='0.001 0.001 0.001')
 ET.SubElement(world,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',rgba='0.65 0.72 0.82 1')
ET.indent(m)
out=root/'biped_static.xml'; ET.ElementTree(m).write(out,encoding='utf-8',xml_declaration=True)
model=mujoco.MjModel.from_xml_path(str(out)); data=mujoco.MjData(model); mujoco.mj_step(model,data)
print('Loaded',len(files),'meshes;',model.ngeom,'geoms; time',data.time)
