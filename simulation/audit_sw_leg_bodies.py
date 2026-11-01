"""Compare both tower parts' visible and hidden solid bodies, read-only."""
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
    result = []
    for comp in doc.GetComponents(False):
        if not comp.GetPathName.lower().endswith('.sldprt') or '16-17-29-589' not in comp.Name2:
            continue
        part = comp.GetModelDoc2
        row = {'component': comp.Name2, 'bodies': {}}
        for mode, visible in [('visible_only', True), ('including_hidden', False)]:
            bodies = part.GetBodies2(0, visible) or []
            details = []
            for body in bodies:
                box = list(body.GetBodyBox())
                massprops = list(body.GetMassProperties(1))
                details.append({'name': body.Name, 'box_size_mm': [(box[i+3]-box[i])*1000 for i in range(3)],
                                'volume_mm3': massprops[3]*1e9})
            row['bodies'][mode] = details
        result.append(row)
    (root / 'sw_leg_bodies_readonly.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for row in result:
        print(row['component'].split('/')[-1], {k: len(v) for k, v in row['bodies'].items()})
        for b in row['bodies']['including_hidden']:
            print(' ', b['name'], [round(v, 2) for v in sorted(b['box_size_mm'])], round(b['volume_mm3'], 2))
finally:
    if doc:
        sw.CloseDoc(doc.GetTitle)
    sw.ExitApp()
    pythoncom.CoUninitialize()
