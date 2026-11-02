import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const root = 'F:/fuxian/Mymodel/output_eval/texturegan_comparison_20261102';
const data = JSON.parse(await fs.readFile(path.join(root,'workbook_data.json'),'utf8'));
if(data.raw.length!==1536) throw new Error('Expected all 1536 metric records');
const wb = Workbook.create();
const main = wb.worksheets.add('指标对比');
const params = wb.worksheets.add('模型参数');
const raw = wb.worksheets.add('逐图数据');
const metrics = ['edge_f1','leak','iou','edge_l1','lpips','tcf_lab'];
const rawKeys = ['arm','id','condition','category',...metrics,'valid_generation','source_file'];
const end = data.raw.length+1;
raw.getRange('A1').write([rawKeys,...data.raw.map(r=>rawKeys.map(k=>r[k]??null))]);
raw.getRange(`A1:L${end}`).format = {font:{name:'Arial',size:10},columnWidth:17,rowHeight:21};
raw.getRange(`L1:L${end}`).format.columnWidth=100;
raw.getRange('A1:L1').format={fill:'#28465d',font:{bold:true,color:'#ffffff'}};
raw.getRange(`E2:J${end}`).setNumberFormat('0.000000');
raw.tables.add(`A1:L${end}`,true,'PerImage128');
raw.freezePanes.freezeRows(1);
raw.freezePanes.freezeColumns(2);
raw.showGridLines=false;

main.showGridLines=false;
main.getRange('A1:H80').format={font:{name:'Arial',size:11},rowHeight:23};
main.getRange('A1:A80').format.columnWidth=29;
main.getRange('B1:B80').format.columnWidth=31;
main.getRange('C1:H80').format.columnWidth=17;
main.getRange('C6:H29').format.horizontalAlignment='right';
main.getRange('A2').values=[['新增128件纯服装 · 12模型指标对比']];
main.getRange('A2').format.font={name:'Arial',size:17,bold:true,color:'#17364e'};
main.getRange('A4:H4').merge();
main.getRange('A4').values=[['仅原参考128件，四类各32；全部生成纳入。均值下列有效n；↑越大越好，↓越小越好。']];
main.getRange('A5:H5').values=[['模型','实际输入','Edge F1 ↑','Leak ↓','IoU ↑','Edge L1 ↓','LPIPS ↓','TCF-LAB ↓']];
main.getRange('A5:H5').format={fill:'#28465d',font:{bold:true,color:'#ffffff'},rowHeight:29};
const checks=[];
data.original.forEach((r,i)=>{
  const y=6+2*i;
  main.getRange(`A${y}:B${y}`).values=[[r.model,r.input_modalities]];
  main.getRange(`A${y+1}`).values=[['有效n']];
  main.getRange(`A${y+1}:H${y+1}`).format={font:{color:'#71808b',size:9},rowHeight:18};
  if(r.model==='TextureGAN') main.getRange(`A${y}:H${y+1}`).format.fill='#fff0dc';
  if(r.model==='T3') main.getRange(`A${y}:H${y+1}`).format.fill='#e8f2fa';
  metrics.forEach((key,j)=>{
    const col=String.fromCharCode(67+j),src=String.fromCharCode(69+j);
    if(r[key]===null){
      main.getRange(`${col}${y}`).values=[['N/A']];
      main.getRange(`${col}${y+1}`).values=[[0]];
      return;
    }
    const criteria=`'逐图数据'!$A$2:$A$${end},"${r.model}",'逐图数据'!$C$2:$C$${end},"original"`;
    main.getRange(`${col}${y}`).formulas=[[`=SUMIFS('逐图数据'!$${src}$2:$${src}$${end},${criteria})/${col}${y+1}`]];
    main.getRange(`${col}${y+1}`).formulas=[[`=COUNTIFS(${criteria},'逐图数据'!$${src}$2:$${src}$${end},"<>")`]];
    main.getRange(`${col}${y}`).setNumberFormat('0.000000');
    main.getRange(`${col}${y+1}`).setNumberFormat('0');
    checks.push({cell:`${col}${y}`,expected:r[key],ncell:`${col}${y+1}`,n:r[key+'_n']});
  });
});
main.getRange('A31:H32').merge();
main.getRange('A31').values=[['本次GPU用量（含取消/失败）：'+(data.resources.jobs.reduce((s,j)=>s+j.actual_gpu_seconds,0)/3600).toFixed(4)+' GPU小时；本次CPU推理与评价，已通过16例GPU指标回放。GPU预算上限1小时；旧11方法账单单列。']];
main.getRange('A31:H32').format={wrapText:true,verticalAlignment:'center',font:{size:10,color:'#536c7d'}};
data.notes.forEach((note,i)=>{
  const y=36+i*3;
  main.getRange(`A${y}:H${y+1}`).merge();
  main.getRange(`A${y}`).values=[[note]];
  main.getRange(`A${y}:H${y+1}`).format={wrapText:true,verticalAlignment:'center',font:{size:10,color:'#536c7d'}};
});
main.freezePanes.freezeRows(5);
main.tabColor='#28465d';

const headers=['模型','实际输入','训练步数','采样/尺度/分辨率','权重与SHA256','适用限制'];
params.getRange('A1').write([headers,...data.parameters.map(r=>[r.model_id,
  data.original.find(m=>m.model===r.model_id).input_modalities,r.training_steps??'',r.inference_config??'',
  (r.checkpoint??'')+'\nSHA256: '+(r.checkpoint_sha256??''),r.comparability_notes??r.limitations??''])]);
params.getRange('A1:F13').format={font:{name:'Arial',size:10},rowHeight:90,wrapText:true,verticalAlignment:'center'};
params.getRange('A1:F1').format={fill:'#28465d',font:{bold:true,color:'#ffffff'},rowHeight:27};
for(const [col,width] of [['A',28],['B',28],['C',24],['D',65],['E',88],['F',65]]) params.getRange(`${col}1:${col}13`).format.columnWidth=width;
params.freezePanes.freezeRows(1);
params.showGridLines=false;
wb.recalculate();
for(const r of checks){
  const actual=main.getRange(r.cell).values[0][0],n=main.getRange(r.ncell).values[0][0];
  if(typeof actual!=='number'||Math.abs(actual-r.expected)>1e-10||n!==r.n) throw new Error('formula mismatch '+JSON.stringify({r,actual,n}));
}
const errorScan=(await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options:{useRegex:true,maxResults:30},summary:'formula error scan'})).ndjson;
await fs.writeFile(path.join(root,'audit/workbook_error_scan.ndjson'),errorScan);
console.log(errorScan);
for(const [sheet,range,file] of [['指标对比','A2:H32','metrics_preview.png'],
  ['模型参数','A1:D4','parameters_preview.png'],['模型参数','A9:F13','parameters_tail_preview.png'],
  ['指标对比','A36:H78','notes_preview.png'],['逐图数据','A1:K8','raw_preview.png']]){
  const png=await wb.render({sheetName:sheet,range,scale:1.4,format:'png'});
  await fs.writeFile(path.join(root,'audit',file),new Uint8Array(await png.arrayBuffer()));
}
const outdir=path.join(root,'outputs','texturegan_comparison_20261102');
await fs.mkdir(outdir,{recursive:true});
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(outdir,'12模型指标与参数.xlsx'));
await fs.writeFile(path.join(root,'audit/workbook_formula_checks.json'),JSON.stringify({passed:true,means_and_n_checked:checks.length},null,2));
console.log('WORKBOOK_EXPORTED');
