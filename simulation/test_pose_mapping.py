import json, numpy as np, win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); docs={d.GetTitle:d for d in s.GetDocuments}; main=docs['SimpleBiped2a_new.SLDASM']; comp={c.Name2:c for c in main.GetComponents(False)}; mates=json.load(open(r'C:\Practice_project\biped_sim\mates.json',encoding='utf8'))
for x in mates:
 if x['type']!='MateConcentric' or len(x['entities'])!=2:continue
 group=x['assembly'];sub=comp[group].GetModelDoc2; subcomps={c.Name2:c for c in sub.GetComponents(False)}; out=[]
 for e in x['entities']:
  name=e['component']; c0=subcomps[name]; c1=comp[group+'/'+name]; t0=np.array(c0.Transform2.ArrayData);t1=np.array(c1.Transform2.ArrayData);R0=t0[:9].reshape(3,3);R1=t1[:9].reshape(3,3);v=np.array(e['parameters'][:3]);u=np.array(e['parameters'][3:6]);
  pt=(v-t0[9:12])@R0.T@R1+t1[9:12]; direction=u@R0.T@R1;out.append((pt,direction))
 p,u=out[0];q,v=out[1];dist=np.linalg.norm(np.cross(q-p,u));parallel=abs(np.dot(u,v)); print(group[-30:],x['name'],'offset mm',round(dist*1000,4),'parallel',round(parallel,6))
