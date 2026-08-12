import argparse, csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from app.defect.defect_bank import DefectExemplarBank
from app.defect.dinov2_adapter import DINOv2Adapter

EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def args_parse():
    p=argparse.ArgumentParser()
    p.add_argument('--test-dir',default='data/screw/test')
    p.add_argument('--ground-truth-dir',default='data/screw/ground_truth')
    p.add_argument('--output-dir',default='outputs/screw/dinov2_gt_roi_3shot')
    p.add_argument('--shots',type=int,default=3)
    p.add_argument('--margin',type=float,default=0.5)
    p.add_argument('--device',default=None)
    return p.parse_args()

def imgs(d):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTS])

def gt_bbox(mask_path):
    m=np.asarray(Image.open(mask_path).convert('L'))
    ys,xs=np.where(m>0)
    if len(xs)==0: raise RuntimeError(f'empty GT mask: {mask_path}')
    return int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1

def crop_roi(image,bbox,margin):
    image=image.convert('RGB'); w,h=image.size
    x1,y1,x2,y2=bbox; bw=max(1,x2-x1); bh=max(1,y2-y1)
    cx=(x1+x2)/2; cy=(y1+y2)/2
    side=int(np.ceil(max(bw*(1+2*margin),bh*(1+2*margin))))
    left=int(np.floor(cx-side/2)); top=int(np.floor(cy-side/2))
    right=left+side; bottom=top+side
    arr=np.asarray(image); border=np.concatenate([arr[0],arr[-1],arr[:,0],arr[:,-1]],axis=0)
    fill=tuple(int(v) for v in np.median(border,axis=0))
    out=Image.new('RGB',(side,side),fill)
    sl,st=max(0,left),max(0,top); sr,sb=min(w,right),min(h,bottom)
    out.paste(image.crop((sl,st,sr,sb)),(sl-left,st-top))
    return out

def main():
    a=args_parse(); test=Path(a.test_dir); gt=Path(a.ground_truth_dir); out=Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    classes=sorted([d for d in test.iterdir() if d.is_dir() and d.name.lower()!='good'])
    ext=DINOv2Adapter(device=a.device); ext.load()
    embs=[]; labels=[]; support_paths=[]; queries=[]
    for d in classes:
        files=imgs(d)
        if len(files)<=a.shots: raise RuntimeError(f'{d.name}: not enough images')
        for i,p in enumerate(files):
            mp=gt/d.name/f'{p.stem}_mask.png'
            roi=crop_roi(Image.open(p),gt_bbox(mp),a.margin)
            split='support' if i<a.shots else 'query'
            rp=out/'roi_images'/split/d.name/p.name; rp.parent.mkdir(parents=True,exist_ok=True); roi.save(rp)
            if i<a.shots:
                embs.append(ext.embed(roi)); labels.append(d.name); support_paths.append(str(rp.resolve()))
            else:
                queries.append((d.name,p,roi,rp))
    bank=DefectExemplarBank(np.stack(embs),labels,support_paths); bank.save(out/'bank')
    rows=[]; total=Counter(); correct=Counter(); conf=defaultdict(Counter)
    for i,(true,p,roi,rp) in enumerate(queries,1):
        r=bank.predict_embedding(ext.embed(roi)); pred=r['predicted_class']; ok=pred==true
        total[true]+=1; correct[true]+=int(ok); conf[true][pred]+=1
        rows.append({'true_class':true,'predicted_class':pred,'correct':int(ok),'image_path':str(p),'roi_path':str(rp),'top1_similarity':r['top1_similarity'],'top2_class':r['top2_class'],'top2_similarity':r['top2_similarity'],'margin':r['margin'],'nearest_exemplar':r['nearest_exemplar']})
        print(f'[{i:03d}/{len(queries):03d}] {true}/{p.name} => {pred} sim={r["top1_similarity"]:.4f} margin={r["margin"]:.4f} {"OK" if ok else "MISS"}')
    with open(out/'fewshot_results.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    names=sorted(set(bank.classes)|set(total))
    with open(out/'confusion_matrix.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['true\\pred',*names]); [w.writerow([t,*[conf[t][p] for p in names]]) for t in names]
    n=sum(total.values()); c=sum(correct.values())
    print('\n=============== GT-ROI 3-shot 汇总 ===============')
    print(f'Top-1 accuracy: {c/max(1,n):.2%} ({c}/{n})')
    for name in names: print(f'{name:<20} {correct[name]:>3}/{total[name]:<3} = {correct[name]/max(1,total[name]):.2%}')
    print(f'results: {(out/"fewshot_results.csv").resolve()}')
    print(f'confusion matrix: {(out/"confusion_matrix.csv").resolve()}')
    print('===================================================')

if __name__=='__main__': main()
