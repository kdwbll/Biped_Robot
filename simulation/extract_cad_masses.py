from pathlib import Path
import json, numpy as np, win32com.client as w
p=Path(r'C:\Practice_project\biped_sim');s=w.GetActiveObject('SldWorks.Application.33');d=next(x for x in s.GetDocuments if x.GetTitle=='SimpleBiped2a_new.SLDASM');cs={c.Name2:c for c in d.GetComponents(False)};sets=list(json.loads((p/'axis_candidates.json').read_text(encoding='utf8')).items());rows=[]
for name,c in cs.items():
 if not c.GetPathName.lower().endswith('.sldprt') or c.IsSuppressed:continue
 found=[i for i,(key,_) in enumerate(sets) if name.startswith(key+'/')];leg=found[0] if found else None;localname=name.split('/')[-1]
 if leg is not None:
  group=sets[leg][0];P=np.array(cs[group].Transform2.ArrayData);Rp=P[:9].reshape(3,3);parts=sorted({v for x in sets[leg][1] for v in x['components']});target=f'L{leg}_part{parts.index(localname)}'
 else: target='robot_free'
 T=np.array(c.Transform2.ArrayData);R=T[:9].reshape(3,3)
 for i,b in enumerate(c.GetModelDoc2.GetBodies2(0,True)):
  box=np.array(b.GetBodyBox());dim=box[3:]-box[:3];pr=np.array(b.GetMassProperties(1));cen=pr[:3]@R+T[9:12];orient=R
  if leg is not None: cen=(cen-P[9:12])@Rp.T;orient=R@Rp.T
  ordered=np.sort(dim)*1000
  is_servo=(any(x in localname for x in ('16-17-29-589','16-18-52-233')) and np.allclose(ordered,[20,47,55],atol=.2)) or ('16-40-03-958' in localname and np.allclose(ordered,[20,43,55],atol=.2))
  mass=.055 if is_servo else pr[3]*1240
  I_local=np.diag([mass*(dim[1]**2+dim[2]**2)/12,mass*(dim[0]**2+dim[2]**2)/12,mass*(dim[0]**2+dim[1]**2)/12]);I=orient.T@I_local@orient
  rows.append({'component':name,'solid':i,'body':target,'kind':'servo_candidate' if is_servo else 'printed_assumed','mass_kg':mass,'volume_m3':pr[3],'center_m':cen.tolist(),'approx_inertia_at_center':I.tolist(),'box_size_m':dim.tolist()})
(p/'cad_mass_inventory.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf8')
print('solids',len(rows),'servo candidates',sum(r['kind']=='servo_candidate' for r in rows),'mass before shafts kg',round(sum(r['mass_kg'] for r in rows),6))
for r in rows:
 if r['kind']=='servo_candidate':print(r['body'],r['component'].split('/')[-1], 'mass',r['mass_kg'])
