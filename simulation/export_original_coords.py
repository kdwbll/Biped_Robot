import os,win32com.client as w
s=w.GetActiveObject('SldWorks.Application.33'); d=s.ActiveDoc
out=r'C:\Practice_project\biped_sim\meshes_original_coords'; os.makedirs(out,exist_ok=True)
old=s.GetUserPreferenceToggle(71); print('original setting:',old)
try:
 print('setting result:',s.SetUserPreferenceToggle(71,True))
 print('export result:',d.SaveAs3(os.path.join(out,'assembly.stl'),0,2))
finally:
 print('restored:',s.SetUserPreferenceToggle(71,old))
print('files',len(os.listdir(out)))
