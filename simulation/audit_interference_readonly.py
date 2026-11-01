"""Recalculate saved assembly interference in an isolated read-only session."""
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
manager = None
try:
    err = w.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    warn = w.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    doc = sw.OpenDoc6(path, 2, 3, '', err, warn)
    if not doc:
        raise RuntimeError(f'OpenDoc6 failed: {err.value}, {warn.value}')
    manager = doc.InterferenceDetectionManager
    manager.TreatCoincidenceAsInterference = False
    entries = manager.GetInterferences or []
    rows = []
    for item in entries:
        row = {'components': [c.Name2 for c in item.Components], 'volume_m3': item.Volume}
        try:
            body = item.GetInterferenceBody
            if callable(body):
                body = body()
            row['interference_body_box_m'] = list(body.GetBodyBox()) if body else None
            row['is_fastener'] = bool(item.IsFastener)
            row['is_possible_interference'] = bool(item.IsPossibleInterference)
        except Exception as exc:
            row['body_error'] = str(exc)
        rows.append(row)
    (root / 'sw_interference_readonly_current.json').write_text(json.dumps(
        {'document': path, 'open_errors': err.value, 'open_warnings': warn.value,
         'coincidence_as_interference': False, 'interferences': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('open errors/warnings:', err.value, warn.value, 'interferences:', len(rows))
    for row in rows:
        print('volume mm3',round(row['volume_m3']*1e9,4),'bounds mm',
              [round(v*1000,3) for v in row.get('interference_body_box_m',[])],row.get('body_error',''))
finally:
    if manager:
        manager.Done()
    if doc:
        sw.CloseDoc(doc.GetTitle)
    sw.ExitApp()
    pythoncom.CoUninitialize()
