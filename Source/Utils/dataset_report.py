"""Report layout and a dependency-free OOXML serializer for the user's runner.

Experimental raw counts stay intact; summary/accuracy formulas include cached
values so reports open correctly without relying on a recalculation engine.
"""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import zipfile
from xml.sax.saxutils import escape

import numpy as np

DETAIL_FIELDS = (
    'dataset','image_id','filename','trial','parameter_value','status',
    'accuracy','ber','correct_bits','total_bits','psnr_embed_db','psnr_attack_db',
    'embed_x','embed_y','embed_offset','extract_x','extract_y','blocks_used',
    'block_size','qr_size','gap','stride','q','payload_seed','phase_seed',
    'layout_seed','extract_seed','noise_seed','seconds','error',
    'psnr_embed_infinite','psnr_attack_infinite',
)
SUMMARY_FIELDS = (
    'dataset','parameter_value','images_attempted','trials_attempted','trials_ok','trials_error',
    'accuracy_min','accuracy_max','accuracy_mean','accuracy_median','accuracy_std',
    'accuracy_p05','accuracy_p95','psnr_embed_mean_db',
)
SUMMARY_HEADERS = (
    'Dataset','Attack level','Images','Trials','Valid trials','Failed trials',
    'Min','Max','Mean','Median','Std. dev.','P5','P95','PSNR (dB)',
)
DETAIL_HEADERS = (
    'Dataset','Image ID','Filename','Trial','Attack level','Status','Correct bits (%)','BER (%)',
    'Correct bits','Total bits','Embedding PSNR (dB)','Attack PSNR (dB)',
    'Embedding x','Embedding y','Spectral offset','Extraction x','Extraction y','Windows used',
    'Block','QR','Gap','Stride','Q','Seed payload','Phase seed','Layout seed',
    'Extraction seed','Noise seed','Time (s)','Error','Infinite embedding PSNR','Infinite attack PSNR',
)


def column_name(index):
    name = ''
    while index:
        index,rem = divmod(index-1,26)
        name = chr(65+rem)+name
    return name


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['attack'],row['dataset'],row['parameter_value'])].append(row)
    summaries = []
    for (attack,size,value),group in sorted(groups.items()):
        valid = [r for r in group if r['status']=='ok']
        accuracy = np.array([r['accuracy'] for r in valid],dtype=np.float64)
        psnr = [r['psnr_embed_db'] for r in valid if r['psnr_embed_db'] is not None]
        summary = dict(attack=attack,dataset=size,parameter_value=value,
                       images_attempted=len({r['image_id'] for r in group}),
                       trials_attempted=len(group),trials_ok=len(valid),trials_error=len(group)-len(valid),
                       accuracy_min=None,accuracy_max=None,accuracy_mean=None,accuracy_median=None,
                       accuracy_std=None,accuracy_p05=None,accuracy_p95=None,
                       psnr_embed_mean_db=float(np.mean(psnr)) if psnr else None)
        if accuracy.size:
            summary.update(accuracy_min=float(accuracy.min()),accuracy_max=float(accuracy.max()),
                           accuracy_mean=float(accuracy.mean()),accuracy_median=float(np.median(accuracy)),
                           accuracy_std=float(accuracy.std(ddof=1)) if accuracy.size>1 else None,
                           accuracy_p05=float(np.percentile(accuracy,5)),accuracy_p95=float(np.percentile(accuracy,95)))
        summaries.append(summary)
    return summaries


def write_csv(path, rows, fields):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def make_report_spec(rows, manifest):
    summaries = summarize(rows)
    sheets = []
    for name in manifest['settings']['attacks']:
        details = sorted([r for r in rows if r['attack']==name],
                         key=lambda r:(r['dataset'],r['parameter_value'],r['image_id'],r['trial']))
        summary = [s for s in summaries if s['attack']==name]
        matrix = [[],[f'Results: {name}'],
                  [f"{manifest['label']} | Placement/noise seed: {manifest['settings']['seed']} | Gap: {manifest['settings']['gap']}"],
                  ['Min-Max and P5-P95 describe the observed trial distribution, grouped by dataset and attack level.'],
                  ['Failed trials are counted separately. Synchronization is not used. The extraction offset is fixed across attack levels.'],
                  [],list(SUMMARY_HEADERS)]
        formulas = []
        summary_start = 8
        for s in summary:
            matrix.append([s[field] for field in SUMMARY_FIELDS])
        matrix += [[],['Per-image results; accuracy = correct bits / total bits. Large seed values are stored as strings to prevent Excel precision loss.'],list(DETAIL_HEADERS)]
        detail_header = len(matrix)
        detail_start = detail_header+1
        bounds = {}
        for index,r in enumerate(details,start=detail_start):
            key=(r['dataset'],r['parameter_value'])
            bounds.setdefault(key,[index,index])[1]=index
            values = [r.get(field) for field in DETAIL_FIELDS]
            for j,field in enumerate(DETAIL_FIELDS):
                if field.endswith('_seed') and values[j] is not None:
                    values[j] = str(values[j])
            matrix.append(values)
            if r['status']=='ok':
                formulas.append({'cell':f'G{index}','formula':f'=I{index}/J{index}','value':r['accuracy']})
                formulas.append({'cell':f'H{index}','formula':f'=1-G{index}','value':r['ber']})
        for index,s in enumerate(summary,start=summary_start):
            first,last = bounds[(s['dataset'],s['parameter_value'])]
            rng=f'G{first}:G{last}'
            formulas += [
                {'cell':f'E{index}','formula':f'=COUNT({rng})','value':s['trials_ok']},
                {'cell':f'F{index}','formula':f'=D{index}-E{index}','value':s['trials_error']},
            ]
            for col,field,expr in (
                ('G','accuracy_min',f'MIN({rng})'),('H','accuracy_max',f'MAX({rng})'),
                ('I','accuracy_mean',f'AVERAGE({rng})'),('J','accuracy_median',f'MEDIAN({rng})'),
                ('K','accuracy_std',f'STDEV({rng})'),('L','accuracy_p05',f'PERCENTILE({rng},0.05)'),
                ('M','accuracy_p95',f'PERCENTILE({rng},0.95)'),
            ):
                minimum=2 if col=='K' else 1
                formulas.append({'cell':f'{col}{index}',
                                 'formula':f'=IF(COUNT({rng})<{minimum},"",{expr})','value':s[field]})
            psnr=f'K{first}:K{last}'
            formulas.append({'cell':f'N{index}','formula':f'=IF(COUNT({psnr})=0,"",AVERAGE({psnr}))',
                             'value':s['psnr_embed_mean_db']})
        sheets.append(dict(name=name,rows=matrix,formulas=formulas,summary_start=summary_start,
                           summary_end=summary_start+len(summary)-1,detail_header=detail_header,
                           detail_start=detail_start,detail_end=len(matrix)))
    return {'sheets':sheets,'summaries':summaries,'label':manifest['label']}


def _cell(ref, value, style=0, formula=None):
    attr=f' r="{ref}" s="{style}"'
    if formula is not None:
        if value is None:
            return f'<c{attr} t="str"><f>{escape(formula.lstrip("="))}</f><v></v></c>'
        return f'<c{attr}><f>{escape(formula.lstrip("="))}</f><v>{value}</v></c>'
    if value is None:
        return ''
    if isinstance(value,bool):
        return f'<c{attr} t="b"><v>{int(value)}</v></c>'
    if isinstance(value,(int,float)) and math.isfinite(value):
        return f'<c{attr}><v>{value}</v></c>'
    value=re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]','',str(value))
    return f'<c{attr} t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>'


def write_xlsx(path, spec):
    """Portable runtime export: only Python stdlib, Excel-compatible formulas/styles."""
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    styles=f'''<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="{ns}">
<numFmts count="2"><numFmt numFmtId="164" formatCode="0.00%"/><numFmt numFmtId="165" formatCode="0.00"/></numFmts>
<fonts count="3"><font><sz val="10"/><name val="Arial"/></font><font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Arial"/></font><font><b/><sz val="14"/><name val="Arial"/></font></fonts>
<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF243B53"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="5"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    content=['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
             '<Default Extension="xml" ContentType="application/xml"/>',
             '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
             '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    workbook_sheets=[]
    workbook_rels=['<Relationship Id="styles" Type="'+rel+'/styles" Target="styles.xml"/>']
    target=Path(path)
    temp=target.with_name(target.name+'.tmp')
    try:
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            for index,sheet in enumerate(spec['sheets'],start=1):
                name=sheet['name']
                if len(name)>31 or re.search(r'[\\/*?:\[\]]',name):
                    raise ValueError(f'Invalid Excel sheet name: {name}')
                workbook_sheets.append(f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>')
                workbook_rels.append(f'<Relationship Id="rId{index}" Type="{rel}/worksheet" Target="worksheets/sheet{index}.xml"/>')
                content.append(f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
                formulas={f['cell']:f for f in sheet['formulas']}
                rows=[]
                for r,values in enumerate(sheet['rows'],start=1):
                    cells=[]
                    for c,value in enumerate(values,start=1):
                        ref=f'{column_name(c)}{r}'
                        style=1 if r in (7,sheet['detail_header']) else 0
                        if r==2: style=4
                        if sheet['summary_start']<=r<=sheet['summary_end'] and 7<=c<=13: style=2
                        if r>=sheet['detail_start'] and c in (7,8): style=2
                        if (r>=sheet['detail_start'] and c in (11,12,23,29)) or (sheet['summary_start']<=r<=sheet['summary_end'] and c==14): style=3
                        f=formulas.get(ref)
                        cells.append(_cell(ref,f['value'] if f else value,style,f['formula'] if f else None))
                    height=' ht="34" customHeight="1"' if r in (7,sheet['detail_header']) else ''
                    rows.append(f'<row r="{r}"{height}>'+''.join(cells)+'</row>')
                cols=[]
                for c in range(1,len(DETAIL_FIELDS)+1):
                    width=20 if c==3 else (32 if c in (26,27,28,30) else 15)
                    cols.append(f'<col min="{c}" max="{c}" width="{width}" customWidth="1"/>')
                xml=f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0" showGridLines="0"><pane ySplit="7" topLeftCell="A8" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols>'+''.join(cols)+'</cols><sheetData>'+''.join(rows)+f'</sheetData><autoFilter ref="A{sheet["detail_header"]}:{column_name(len(DETAIL_FIELDS))}{sheet["detail_end"]}"/></worksheet>'
                z.writestr(f'xl/worksheets/sheet{index}.xml',xml)
            z.writestr('[Content_Types].xml','<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'+''.join(content)+'</Types>')
            z.writestr('_rels/.rels',f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="root" Type="{rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
            z.writestr('xl/workbook.xml',f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{ns}" xmlns:r="{rel}"><bookViews><workbookView/></bookViews><sheets>'+''.join(workbook_sheets)+'</sheets><calcPr calcId="191029" fullCalcOnLoad="1"/></workbook>')
            z.writestr('xl/_rels/workbook.xml.rels','<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(workbook_rels)+'</Relationships>')
            z.writestr('xl/styles.xml',styles)
        temp.replace(target)
    finally:
        if temp.exists(): temp.unlink()


def export_reports(run_dir, rows, manifest, json_only=False):
    run_dir=Path(run_dir)
    spec=make_report_spec(rows,manifest)
    fields=['attack',*DETAIL_FIELDS]
    write_csv(run_dir/'details.csv',rows,fields)
    write_csv(run_dir/'summary.csv',spec['summaries'],['attack',*SUMMARY_FIELDS])
    (run_dir/'report_spec.json').write_text(json.dumps(spec,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    if not json_only:
        write_xlsx(run_dir/'dataset_results.xlsx',spec)
    return spec
