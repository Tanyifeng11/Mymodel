"""O4 完整生成方向跟随：E5 / S0 / S0+O4，独立参考，零训练。"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from garment_mask_utils import build_sketch_garment_mask, mask_backend_info
from models.localized_spatial import LocalizedAdapter, LocalizedInjection
from models.pattern_canonicalization import fft_orientation, rotation_only
from models.spatial_geometry import geometry_map
from tools.e15_common import write_json
from tools.e15_d5_generation import build_inference_args, load_inference_module
from tools.e18_1_clean_patterns import make_image, PALETTES
from tools.e19_handcrafted import bf_tokens, rgb, text_tokens
from tools.e20_utilization import case_stat, frozen_digest, load_pipeline
from tools.e22_o4_metrics import axial_distance, fixed_roi, measure, orientation


SEEDS = (42,43)
ARMS = ("E5","S0","O4")


def sha(value):
    return hashlib.sha256(value).hexdigest()


class ConditionalO4(LocalizedInjection):
    def __init__(self, unet, adapter, site):
        self.active, self.conditional, self.unconditional = False, 0, 0
        super().__init__(unet,adapter,site)
        self.pre = unet.register_forward_pre_hook(self.before,with_kwargs=True)

    def before(self,module,args,kwargs):
        self.active = "sa_hidden_states" in (kwargs.get("cross_attention_kwargs") or {})
        if self.active:
            self.conditional += 1
        else:
            self.unconditional += 1

    def _inject(self,module,inputs,output):
        return super()._inject(module,inputs,output) if self.active else output

    def close(self):
        self.pre.remove()
        super().close()


def prepare_cases(root,data_root,out,width,height):
    path = out/"cases.json"
    if path.exists():
        return json.loads(path.read_text())
    folder = out/"inputs"
    folder.mkdir(exist_ok=True)
    old = json.loads((root/"e19_2_b/data/manifest.json").read_text())["splits"]
    old_hashes = {r["sha256"] for s in ("train","primary") for r in old[s]}
    used_freq = {r["frequency"] for s in ("train","primary") for r in old[s]}
    refs = []
    for i,(f,p) in enumerate((f,p) for f in (6,7,9,11) for p in (.17,.39)):
        assert f not in used_freq
        image = make_image(f,p,PALETTES[i%4])
        if i%2:
            image = image.transpose(Image.Transpose.ROTATE_90)
        pair = []
        for variant,im in (("original",image),("rot90",image.transpose(Image.Transpose.ROTATE_90))):
            digest = sha(im.tobytes())
            assert digest not in old_hashes
            name = "inputs/ref%02d_%s.png" % (i,variant)
            im.save(out/name)
            pair.append({"variant":variant,"path":name,"sha256":digest,
                         "theta":float((90 if i%2==0 else 0)+(90 if variant=="rot90" else 0))%180})
        refs.append({"id":i,"frequency":f,"phase":p,"palette":i%4,"variants":pair})
    previous = json.loads((root/"e20/cache_manifest.json").read_text())
    used_sketches = {r["hash"] for r in previous["train_sketches"]}
    used_sketches.update(sha(Image.open(root/("e19_2_c/cases/%02d_sketch.png"%i)).convert("RGB").tobytes()) for i in range(32))
    rows = json.loads((root/"data/processed/bf_full_audit_v1/validation_clean.json").read_text())
    sketches, masks = [], []
    for index,row in enumerate(rows):
        im = Image.open(data_root/row["sketch"]).convert("RGB").resize((width,height),Image.BILINEAR)
        digest = sha(im.tobytes())
        if digest in used_sketches or any(digest==s["sha256"] for s in sketches):
            continue
        mask,info = build_sketch_garment_mask(im,width,height)
        m = np.array(mask)>127
        roi = fixed_roi(mask)
        if not .08<float(m.mean())<.85 or roi is None or roi[2]<96:
            continue
        if masks and (m&masks[0]).sum()/max((m|masks[0]).sum(),1)>.7:
            continue
        j = len(sketches)
        im.save(folder/("sketch%d.png"%j))
        mask.save(folder/("mask%d.png"%j))
        sketches.append({"id":j,"manifest_index":index,"source":row["sketch"],"sha256":digest,
                         "path":"inputs/sketch%d.png"%j,"mask":"inputs/mask%d.png"%j,"roi":roi,"mask_info":info})
        masks.append(m)
        if len(sketches)==2:
            break
    assert len(sketches)==2, "Need two eligible independent garment silhouettes"
    calibration = []
    for r in refs:
        for v in r["variants"]:
            image = Image.open(out/v["path"]).resize((width,height),Image.BILINEAR)
            for s in sketches:
                result = orientation(image,s["roi"])
                calibration.append({"reference":r["id"],"variant":v["variant"],"sketch":s["id"],**result})
                assert result["valid"] and axial_distance(result["theta"],v["theta"])<=20
    result = {"references":refs,"sketches":sketches,"calibration":calibration,
              "reference_selection":"four unseen O4 frequencies x two unseen phases; exact originals and true pixel rot90; balanced starting axes",
              "sketch_selection":"ascending validation manifest, excludes all O4 train/eval sketch hashes; valid mask and ROI only; pair mask IoU<=0.7",
              "synthetic_references":True,"selection_uses_generated_images":False}
    write_json(path,result)
    return result


def e5_pipeline(root):
    checkpoint = str(root/"output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt")
    args = argparse.Namespace(checkpoint=checkpoint,texture_ckpt=checkpoint,
                              base_model_path=str(root/"models/stable-diffusion-v1-5"),
                              vae_model_path=str(root/"models/stable-diffusion-v1-5/vae"),
                              clip_model=str(root/"models/clip"),device="cuda:0",seed=42,steps=50)
    ns = build_inference_args(args)
    ns.texture_num_tokens, ns.force_texture_num_tokens_override = 16,False
    pipe,_ = load_inference_module().prepare(ns)
    modules = {k:getattr(pipe,k) for k in ("unet","reference_unet","bf_texture_conditioner","tcpm_lite","vae","text_encoder","image_encoder")}
    for m in modules.values():
        m.eval().requires_grad_(False)
    return pipe,modules,ns.width,ns.height


@torch.no_grad()
def current_tokens(pipe,bf,pattern,image,width,height):
    text = text_tokens(pipe,["a cloth"])
    app = bf_tokens(pipe,bf,image,text,width,height)
    source = rgb(image,pipe.device)
    identity = pattern.identity_tokens(rotation_only(source,fft_orientation(source)[0])).half()
    geo = pattern.geometry_tokens(source).half()
    return pipe.tcpm_lite(torch.cat([app,identity,geo],1),text)


@torch.no_grad()
def current_null(pipe,bf,width,height):
    negative = text_tokens(pipe,[" worst quality, low quality"])
    pixels = pipe.clip_image_processor(images=[Image.new("RGB",(256,256))],return_tensors="pt").pixel_values
    vision = pipe.image_encoder(pixels.to(pipe.device,torch.float16),output_hidden_states=True)
    app = bf(clip_image_embeds=torch.zeros_like(vision.image_embeds),
             clip_vision_tokens=torch.zeros_like(vision.hidden_states[-1][:,1:]),
             texture_images=torch.zeros((1,3,height,width),device=pipe.device,dtype=torch.float16),
             texture_mode="patch_resampled",text_embeds=negative,
             apply_text_guidance=True,apply_film=True,apply_nexus=True)[0]
    # No reference enters CFG negative: current BF zero-input + eight null pattern tokens.
    return pipe.tcpm_lite(torch.cat([app,torch.zeros_like(app[:,:8])],1),negative)


def generate_arm(pipe,arm,cases,out,width,height,bank=None,injection=None):
    folder = out/arm
    folder.mkdir(exist_ok=True)
    rows = []
    saved_get, saved_latents = pipe.get_image_embeds,pipe.prepare_latents
    latent_hash = {}
    def latents(self,*args,**kwargs):
        value = saved_latents(*args,**kwargs)
        latent_hash["value"] = sha(value.detach().cpu().contiguous().numpy().tobytes())
        return value
    pipe.prepare_latents = MethodType(latents,pipe)
    pipe.set_progress_bar_config(disable=True)
    for ref in cases["references"]:
        for sk in cases["sketches"]:
            sketch = Image.open(out/sk["path"]).convert("RGB")
            mask_image = Image.open(out/sk["mask"]).convert("L")
            mask = to_tensor(mask_image)[None].to(pipe.device,torch.float16)
            for seed in SEEDS:
                for v in ref["variants"]:
                    key = (ref["id"],v["variant"])
                    image = Image.open(out/v["path"]).convert("RGB")
                    name = "r%02d_k%d_s%d_%s"%(ref["id"],sk["id"],seed,v["variant"])
                    meta_path = folder/(name+".json")
                    if meta_path.exists():
                        rows.append(json.loads(meta_path.read_text()))
                        continue
                    if bank is not None:
                        pair = bank[key]
                        pipe.get_image_embeds = MethodType(lambda self,**kwargs:pair,pipe)
                    if injection is not None:
                        # Reproduce E22/O4's 32x32 map cache before feature interpolation.
                        geo = F.interpolate(geometry_map(rgb(image,pipe.device)),(32,32),mode="area")[:,[0,1,4]]
                        injection.set(geo.half(),mask)
                        injection.conditional = injection.unconditional = 0
                    with torch.inference_mode():
                        generated = pipe(prompt="a cloth",null_prompt="",negative_prompt=" worst quality, low quality",
                                         ref_image=(to_tensor(sketch)[None]*2-1),texture_clip_image=image,
                                         width=width,height=height,num_inference_steps=50,guidance_scale=7.,
                                         sketch_scale=.6,ipa_scale=1.,texture_mode="patch_resampled",
                                         texture_condition_mode="token",texture_preprocess_mode="plain_resize",
                                         texture_num_tokens=16 if arm=="E5" else 24,
                                         force_texture_num_tokens_override=arm!="E5",texture_scale=1.,
                                         spatial_mask=mask,generator=torch.Generator(device=pipe.device).manual_seed(seed))[0]
                    generated.save(folder/(name+".png"))
                    row = {"arm":arm,"reference":ref["id"],"sketch":sk["id"],"seed":seed,"variant":v["variant"],
                           "path":str((folder/(name+".png")).relative_to(out)),"initial_latent_sha256":latent_hash["value"],
                           **measure(generated,mask_image,sketch,image,v["theta"],sk["roi"])}
                    if injection is not None:
                        assert injection.conditional==50 and injection.unconditional==50
                        row["o4_conditional_calls"] = injection.conditional
                        row["o4_unconditional_calls_skipped"] = injection.unconditional
                    write_json(meta_path,row)
                    rows.append(row)
                    print("[o4-generation]",arm,len(rows),"/64",name,row["direction"],flush=True)
        write_json(folder/"records_partial.json",rows)
    pipe.get_image_embeds,pipe.prepare_latents = saved_get,saved_latents
    write_json(folder/"records.json",rows)
    return rows


def summarize(records,out):
    assert all(len(records[arm])==64 for arm in ARMS)
    pairs,groups = {},{}
    for arm,rows in records.items():
        index = {(r["reference"],r["sketch"],r["seed"],r["variant"]):r for r in rows}
        arm_pairs = []
        for k,a in index.items():
            if k[-1]!="original":
                continue
            b = index[k[:3]+("rot90",)]
            delta = axial_distance(a["direction"]["theta"],b["direction"]["theta"])
            valid = a["direction"]["valid"] and b["direction"]["valid"]
            arm_pairs.append({"reference":k[0],"sketch":k[1],"seed":k[2],
                              "flip_correct":a["direction"]["correct"] and b["direction"]["correct"] and delta>=70,
                              "delta_theta":delta,"valid_delta_theta":delta if valid else 0.,"valid_pair":valid,
                              "rotation_color_delta":float(np.linalg.norm(np.array(a["interior_lab"])-b["interior_lab"]))})
        pairs[arm] = arm_pairs
        stat = lambda key:case_stat([(r["reference"],float(r[key])) for r in arm_pairs])
        groups[arm] = {"direction_accuracy":case_stat([(r["reference"],float(r["direction"]["correct"])) for r in rows]),
                       "direction_valid_fraction":case_stat([(r["reference"],float(r["direction"]["valid"])) for r in rows]),
                       **{k:stat(k) for k in ("flip_correct","delta_theta","valid_delta_theta","rotation_color_delta")},
                       **{k:case_stat([(r["reference"],r[k]) for r in rows]) for k in
                          ("sketch_iou","edge_f1","contour_f1","leakage","background_white_mae","reference_color_delta")},
                       "by_seed":{str(s):float(np.mean([r["flip_correct"] for r in arm_pairs if r["seed"]==s])) for s in SEEDS},
                       "by_sketch":{str(s):float(np.mean([r["flip_correct"] for r in arm_pairs if r["sketch"]==s])) for s in (0,1)}}
    contrasts = {}
    for arm,base in (("O4","S0"),("S0","E5"),("O4","E5")):
        pa,pb = pairs[arm],pairs[base]
        assert [(r["reference"],r["sketch"],r["seed"]) for r in pa]==[(r["reference"],r["sketch"],r["seed"]) for r in pb]
        result = {k:case_stat([(a["reference"],float(a[k])-float(b[k])) for a,b in zip(pa,pb)])
                  for k in ("flip_correct","valid_delta_theta","rotation_color_delta")}
        aa,bb = records[arm],records[base]
        for k in ("sketch_iou","edge_f1","contour_f1","leakage","background_white_mae"):
            result[k] = case_stat([(a["reference"],a[k]-b[k]) for a,b in zip(aa,bb)])
        for label,k in (("color_delta","interior_lab"),("dominant_color_delta","interior_median_lab")):
            result[label] = case_stat([(a["reference"],float(np.linalg.norm(np.array(a[k])-b[k]))) for a,b in zip(aa,bb)])
        result["preservation_pass"] = (all(result[k]["ci95"][0]>=-.02 for k in ("sketch_iou","edge_f1","contour_f1"))
                                       and result["leakage"]["ci95"][1]<=.02 and result["background_white_mae"]["ci95"][1]<=.01
                                       and all(result[k]["ci95"][1]<=3. for k in ("color_delta","dominant_color_delta")))
        contrasts[arm+"_vs_"+base] = result
    following = (groups["O4"]["direction_accuracy"]["mean"]>=.75 and groups["O4"]["flip_correct"]["mean"]>=.5
                 and contrasts["O4_vs_S0"]["flip_correct"]["ci95"][0]>0 and contrasts["O4_vs_S0"]["valid_delta_theta"]["ci95"][0]>0)
    for seed in SEEDS:
        assert len({r["initial_latent_sha256"] for rows in records.values() for r in rows if r["seed"]==seed})==1
    result = {"groups":groups,"contrasts":contrasts,"direction_following_pass":following,
              "preservation_pass":contrasts["O4_vs_S0"]["preservation_pass"],
              "pass":following and contrasts["O4_vs_S0"]["preservation_pass"],
              "count":sum(map(len,records.values())),"initial_noise_identical":True,
              "bootstrap_unit":"8 reference groups, aggregate both sketches/seeds/rotations before bootstrap",
              "scope":"controlled synthetic stripes; no claim of real-reference transfer or broad superiority to E5",
              "period_branch_used":False,"joint_used":False,"training_steps":0}
    write_json(out/"pairs.json",pairs)
    write_json(out/"report.json",result)
    print("[o4-final]",json.dumps(result),flush=True)


def sheets(cases,out):
    folder = out/"previews"
    folder.mkdir(exist_ok=True)
    for sk in cases["sketches"]:
        for seed in SEEDS:
            sheet = Image.new("RGB",(8*128,8*184),"white")
            draw = ImageDraw.Draw(sheet)
            for ref in cases["references"]:
                i = ref["id"]
                sources = [out/v["path"] for v in ref["variants"]]
                sources += [out/arm/("r%02d_k%d_s%d_%s.png"%(i,sk["id"],seed,v["variant"])) for arm in ARMS for v in ref["variants"]]
                for col,path in enumerate(sources):
                    im = Image.open(path).convert("RGB"); im.thumbnail((128,160))
                    sheet.paste(im,(col*128,i*184))
                    draw.text((col*128+2,i*184+161),(["ref","ref90","E5","E5 rot","S0","S0 rot","O4","O4 rot"][col])+" r%d"%i,fill="black")
            sheet.save(folder/("sketch%d_seed%d.jpg"%(sk["id"],seed)),quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    parser.add_argument("--data-root",required=True)
    args = parser.parse_args()
    root,out = Path(args.root),Path(args.root)/"e22_o4_generation"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(42)
    assert json.loads((root/"e22_1/report.json").read_text())["pass"]
    safe = torch.load(root/"e22_1/safe_orientation.pt",map_location="cpu",weights_only=False)
    protocol = {"arms":ARMS,"references":8,"sketches":2,"rotations":2,"seeds":SEEDS,"total_images":192,
                "sampling":"original E5 pipeline/scheduler, 50 steps, CFG7, sketch0.6, texture1, same initial noise",
                "causal_comparison":"S0 vs O4; E5 restored with its original BF and TCPM, 16 tokens; S0/O4 current24 tokens",
                "rotation_intervention":"recompute all positive reference branches for actual pixel rot90; S0/O4 share the exact same token bank; only O4 adds the matching orientation map",
                "current_negative":"current BF zero CLIP/patch/CNN input + eight zero pattern tokens before current TCPM; fixed for every reference and same for S0/O4",
                "adapter":"exact frozen safe_orientation weights, conditional CFG only, all50 timesteps, no strength or timestep search",
                "direction_gate":"O4 direction accuracy>=.75, paired flip>=.50; reference-cluster CI of paired flip and valid angular-response improvement over S0 both strictly positive",
                "ambiguity":"contrast>=.02, tensor coherence>=.25, FFT coherence>=.35, radial concentration>=.20, tensor/FFT disagreement<=20deg; invalid counts as failure",
                "preservation_gate":"O4-S0 IoU/EdgeF1/contourF1 lowerCI>=-.02, leakage upperCI<=.02, background white MAE upperCI<=.01, mean/median interior Lab drift upperCI<=3",
                "region":"same maximal 3:4 rectangle within 17px eroded garment interior; native pixel aspect for orientation",
                "safety_scope":"white-background full-generation proxies, not paired target MSE from E22 denoising",
                "selection":"fixed inputs before any generation; no output-based sample or checkpoint selection",
                "period_branch":False,"training_steps":0,"mask_backend":mask_backend_info()}
    write_json(out/"protocol.json",protocol)
    records,freezes = {},{}
    pipe,modules,width,height = e5_pipeline(root)
    cases = prepare_cases(root,Path(args.data_root),out,width,height)
    protocol.update(width=width,height=height,scheduler=pipe.scheduler.__class__.__name__,scheduler_config=dict(pipe.scheduler.config))
    write_json(out/"protocol.json",protocol)
    before = {k:frozen_digest(m) for k,m in modules.items()}
    records["E5"] = generate_arm(pipe,"E5",cases,out,width,height)
    freezes["E5"] = all(frozen_digest(m)==before[k] for k,m in modules.items())
    del pipe,modules; gc.collect(); torch.cuda.empty_cache()
    pipe,bf,pattern,modules,w,h = load_pipeline(root,"cuda:0")
    assert (w,h)==(width,height) and dict(pipe.scheduler.config)==protocol["scheduler_config"]
    before = {k:frozen_digest(m) for k,m in modules.items()}
    bank = {}
    null = current_null(pipe,bf,width,height)
    assert null.shape[1]==24
    for ref in cases["references"]:
        for v in ref["variants"]:
            bank[(ref["id"],v["variant"])] = (current_tokens(pipe,bf,pattern,Image.open(out/v["path"]).convert("RGB"),width,height),null)
    records["S0"] = generate_arm(pipe,"S0",cases,out,width,height,bank)
    adapter = LocalizedAdapter(channels=640,input_channels=3,policy=safe["policy"]).to(pipe.device)
    adapter.load_state_dict(safe["adapter"]); adapter.eval().requires_grad_(False)
    adapter_before = frozen_digest(adapter)
    injection = ConditionalO4(pipe.unet,adapter,safe["site"])
    records["O4"] = generate_arm(pipe,"O4",cases,out,width,height,bank,injection)
    injection.close()
    freezes["S0_O4"] = all(frozen_digest(m)==before[k] for k,m in modules.items())
    freezes["orientation_adapter"] = frozen_digest(adapter)==adapter_before
    assert all(freezes.values())
    write_json(out/"freeze_audit.json",freezes)
    summarize(records,out); sheets(cases,out)


if __name__ == "__main__":
    main()
