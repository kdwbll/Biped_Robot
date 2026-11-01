"""Read mate metadata from the saved assembly without changing CAD files."""
from pathlib import Path
import json
import pythoncom
import win32com.client as w

root = Path(__file__).parent
path = r'C:\Practice_project\fangzhen\SimpleBiped2a_new.SLDASM'
pythoncom.CoInitialize()
sw = w.DispatchEx('SldWorks.Application.33')
sw.Visible = False
doc = None
try:
    err = w.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    warn = w.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    doc = sw.OpenDoc6(path, 2, 3, '', err, warn)
    if not doc:
        raise RuntimeError(f'OpenDoc6 failed: {err.value}, {warn.value}')
    assemblies = [(doc.GetTitle, doc)]
    for c in doc.GetComponents(False):
        if c.GetPathName.lower().endswith('.sldasm'):
            model = c.GetModelDoc2
            if model:
                assemblies.append((c.Name2, model))
    rows = []
    for assembly, model in assemblies:
        feature = model.FirstFeature
        while feature:
            if feature.GetTypeName2 == 'MateGroup':
                sub = feature.GetFirstSubFeature
                while sub:
                    row = {'assembly': assembly, 'name': sub.Name, 'type': sub.GetTypeName2, 'entities': []}
                    try:
                        mate = sub.GetSpecificFeature2
                        for i in range(mate.GetMateEntityCount):
                            ent = mate.MateEntity(i)
                            ref = ent.ReferenceComponent
                            row['entities'].append({'component': ref.Name2 if ref else None, 'parameters': list(ent.EntityParams)})
                    except Exception as exc:
                        row['error'] = str(exc)
                    rows.append(row)
                    sub = sub.GetNextSubFeature
            feature = feature.GetNextFeature
    result = {'document': path, 'requested_open_options': 3, 'open_errors': err.value, 'open_warnings': warn.value,
              'assemblies': [a for a, _ in assemblies], 'mates': rows}
    (root / 'sw_mates_readonly_current.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Assemblies:', len(assemblies), 'mates:', len(rows), 'errors:', sum('error' in r for r in rows))
    for assembly, _ in assemblies:
        print(assembly, sum(r['assembly'] == assembly for r in rows), 'mates')
finally:
    if doc:
        sw.CloseDoc(doc.GetTitle)
    sw.ExitApp()
    pythoncom.CoUninitialize()
