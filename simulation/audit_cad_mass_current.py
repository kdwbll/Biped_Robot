"""Regenerate a separate, current CAD mass inventory from read-only assembly."""
from pathlib import Path
import json
import numpy as np
import pythoncom
import win32com.client as w

p = Path(__file__).parent
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
    components = {c.Name2: c for c in doc.GetComponents(False)}
    groups = list(json.loads((p / 'axis_candidates.json').read_text(encoding='utf8')).items())
    rows = []
    for name, comp in components.items():
        if not comp.GetPathName.lower().endswith('.sldprt') or comp.IsSuppressed:
            continue
        found = [i for i, (group, _) in enumerate(groups) if name.startswith(group + '/')]
        leg = found[0] if found else None
        localname = name.split('/')[-1]
        if leg is not None:
            group = groups[leg][0]
            transform = np.array(components[group].Transform2.ArrayData)
            parent_rot = transform[:9].reshape(3, 3)
            parts = sorted({v for x in groups[leg][1] for v in x['components']})
            target = f'L{leg}_part{parts.index(localname)}'
        else:
            target = 'robot_free'
        transform_part = np.array(comp.Transform2.ArrayData)
        rot = transform_part[:9].reshape(3, 3)
        for index, body in enumerate(comp.GetModelDoc2.GetBodies2(0, True) or []):
            box = np.array(body.GetBodyBox())
            dim = box[3:] - box[:3]
            props = np.array(body.GetMassProperties(1))
            center = props[:3] @ rot + transform_part[9:12]
            orient = rot
            if leg is not None:
                center = (center - transform[9:12]) @ parent_rot.T
                orient = rot @ parent_rot.T
            ordered = np.sort(dim) * 1000
            servo = (any(x in localname for x in ('16-17-29-589', '16-18-52-233')) and
                     abs(ordered[0]-20)<.2 and 45.8 <= ordered[1] <= 47.2 and abs(ordered[2]-55)<.2) or (
                     '16-40-03-958' in localname and np.allclose(ordered, [20, 43, 55], atol=.2))
            mass = .055 if servo else props[3] * 1240
            inertial = np.diag([mass*(dim[1]**2+dim[2]**2)/12, mass*(dim[0]**2+dim[2]**2)/12,
                                mass*(dim[0]**2+dim[1]**2)/12])
            inertia = orient.T @ inertial @ orient
            rows.append({'component': name, 'solid': index, 'solid_name': body.Name, 'body': target,
                         'kind': 'servo_candidate' if servo else 'printed_assumed', 'mass_kg': mass,
                         'volume_m3': float(props[3]), 'center_m': center.tolist(),
                         'approx_inertia_at_center': inertia.tolist(), 'box_size_m': dim.tolist()})
    if sum(r['kind'] == 'servo_candidate' for r in rows) != 5:
        raise RuntimeError('Expected five servo candidates; inspect geometry before saving the inventory')
    (p/'cad_mass_inventory_current.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
    print('Open errors/warnings', err.value, warn.value, 'solids', len(rows),
          'servo candidates', sum(r['kind'] == 'servo_candidate' for r in rows),
          'mass kg', round(sum(r['mass_kg'] for r in rows), 6))
    for name in ('L0_part0', 'L1_part0'):
        selected = [r for r in rows if r['body'] == name]
        print(name, len(selected), 'solids', round(sum(r['mass_kg'] for r in selected), 6), 'kg')
finally:
    if doc:
        sw.CloseDoc(doc.GetTitle)
    sw.ExitApp()
    pythoncom.CoUninitialize()
