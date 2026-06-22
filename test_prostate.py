"""Test DFG models on PROSTATE dataset (volume-level Dice + ASSD)."""
import argparse, os, numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import get_model
from dataloaders import ProstateDataset
from utils.metrics import MultiDiceScore, MultiASD

ORGAN_LIST = ['Prostate']
NUM_CLASSES = 2
DATA_ROOT = '/opt/data/private/MedSeg_Data_Process/PROSTATE/processed_new'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--domain', type=str, required=True)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--data_root', type=str, default=DATA_ROOT)
    p.add_argument('--img_size', type=int, nargs=2, default=[256, 256])
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu_id}')

    dataset = ProstateDataset(
        data_root=args.data_root,
        domain_name=args.domain,
        phase='val',
        split_train=False,
        img_size=tuple(args.img_size),
    )
    print(f'Domain: {args.domain}, slices: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    cfg = {'arch': 'UNet', 'input_dim': 3, 'num_classes': NUM_CLASSES}
    model = get_model(cfg)
    ckpt = torch.load(args.model_path, map_location='cpu')
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device).eval()

    sample_dict = {}
    with torch.no_grad():
        for imgs, segs, names in tqdm(loader, desc='Inference'):
            imgs = imgs.to(device)
            _, preds = model(imgs)
            for i, name in enumerate(names):
                parts = name.rsplit('_', 1)
                pid, sidx = parts[0], int(parts[1])
                sample_dict.setdefault(pid, []).append(
                    (preds[i].cpu(), segs[i].cpu(), sidx))

    pred_vols, gt_vols = [], []
    patient_ids = []
    for pid in sorted(sample_dict.keys()):
        slices = sorted(sample_dict[pid], key=lambda x: x[2])
        ps, ts = [], []
        for pred, tgt, _ in slices:
            if tgt.sum() == 0:
                continue
            ps.append(pred)
            ts.append(tgt)
        if not ps:
            continue
        pred_vols.append(torch.stack(ps, dim=-1))
        gt_vols.append(torch.stack(ts, dim=-1))
        patient_ids.append(pid)

    print(f'Patients evaluated: {len(pred_vols)}')

    all_dice = np.full((len(pred_vols), 1), np.nan)
    all_assd = np.full((len(pred_vols), 1), np.nan)

    for idx, (pv, gv) in enumerate(zip(pred_vols, gt_vols)):
        gv = gv.long()
        dl = MultiDiceScore(pv, gv, NUM_CLASSES, include_bg=False)
        pv_argmax = pv.argmax(dim=0).long()
        for c, d in enumerate(dl):
            if not np.isnan(d):
                all_dice[idx, c] = d
        try:
            al = MultiASD(pv_argmax, gv, NUM_CLASSES, include_bg=False)
            for c, a in enumerate(al):
                all_assd[idx, c] = a
        except Exception as e:
            print(f'  ASSD warning patient {patient_ids[idx]}: {e}')

    print()
    print('=' * 60)
    print(f'{"Class":<15} {"Dice":>10} {"ASSD":>10}')
    print('-' * 60)
    for c, organ in enumerate(ORGAN_LIST):
        dm = np.nanmean(all_dice[:, c])
        am = np.nanmean(all_assd[:, c])
        print(f'{organ:<15} {dm:>10.4f} {am:>10.4f}')
    md = np.nanmean(all_dice)
    ma = np.nanmean(all_assd)
    print('-' * 60)
    print(f'{"Mean":<15} {md:>10.4f} {ma:>10.4f}')
    print('=' * 60)

    # per-patient
    print()
    print('Per-patient results:')
    for idx, pid in enumerate(patient_ids):
        d = all_dice[idx, 0]
        a = all_assd[idx, 0]
        print(f'  {pid}: Dice={d:.4f}, ASSD={a:.4f}')

if __name__ == '__main__':
    main()
