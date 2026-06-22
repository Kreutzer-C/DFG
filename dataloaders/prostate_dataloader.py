import torch
import torch.utils.data as data
import os
import json
import numpy as np
from .transformations import get_transform, get_transform_strong_Weak


class ProstateDataset(data.Dataset):
    """Dataset for PROSTATE .npz files with metadata.json splits."""

    def __init__(self, data_root, domain_name, phase='train', split_train=True,
                 img_size=(256, 256), weak_strong_aug=False):
        self.data_root = data_root
        self.domain_name = domain_name
        self.phase = phase
        self.weak_strong_aug = weak_strong_aug
        self.img_size = img_size

        if self.weak_strong_aug:
            self.augmenter_w, self.augmenter_s = get_transform_strong_Weak(self.phase, New_size=img_size)
        else:
            self.augmenter = get_transform(self.phase, New_size=img_size)

        with open(os.path.join(data_root, 'metadata.json')) as f:
            metadata = json.load(f)

        split_key = 'train' if split_train else 'test'
        case_ids = metadata['splits'][domain_name][split_key]

        slices_dir = os.path.join(data_root, domain_name, 'slices')
        all_files = sorted(os.listdir(slices_dir))

        self.all_data_path = []
        self.name_list = []
        self.filepath_map = {}

        for fname in all_files:
            if not fname.endswith('.npz'):
                continue
            base = fname[:-4]
            parts = base.split('_slice_')
            if len(parts) != 2:
                continue
            vol_part = parts[0]
            slice_str = parts[1]
            case_id = vol_part.replace('vol_', '')

            if case_id not in case_ids:
                continue

            safe_case = case_id.replace('_', '-')
            slice_idx = int(slice_str)
            name = '{}_{}'.format(safe_case, slice_idx)

            self.all_data_path.append(os.path.join(slices_dir, fname))
            self.name_list.append(name)
            self.filepath_map[name] = os.path.join(slices_dir, fname)

    def __getitem__(self, index):
        d = np.load(self.all_data_path[index])
        name = self.name_list[index]
        img = d['img'].astype(np.float32)
        seg = d['label']

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        if self.weak_strong_aug:
            transformed_w = self.augmenter_w(image=img, mask=seg)
            img_w = transformed_w['image']
            seg = transformed_w['mask']
            transformed_s = self.augmenter_s(image=img_w.numpy().transpose((1, 2, 0)))
            img_s = transformed_s['image']
            return img_w, img_s, seg, name
        else:
            transformed = self.augmenter(image=img, mask=seg)
            img = transformed['image']
            img = img.to(torch.float32)
            seg = transformed['mask']
            seg = seg.to(torch.long)
            return img, seg, name

    def __len__(self):
        return len(self.all_data_path)


class ProstateDataset_refine(data.Dataset):
    """Refined pseudo-label dataset for PROSTATE (SAM stage)."""

    def __init__(self, datadir, phase='train', weak_strong_aug=False, img_size=(256, 256)):
        self.datadir = datadir
        self.phase = phase
        self.weak_strong_aug = weak_strong_aug

        if self.weak_strong_aug:
            self.augmenter_w, self.augmenter_s = get_transform_strong_Weak(self.phase, New_size=img_size)
        else:
            self.augmenter = get_transform(self.phase, New_size=img_size)

        self.all_data_path = []
        self.name_list = []

        for data_name in sorted(os.listdir(datadir)):
            if not data_name.endswith('.npz'):
                continue
            self.name_list.append(data_name[:-4])
            self.all_data_path.append(os.path.join(datadir, data_name))

    def __getitem__(self, index):
        d = np.load(self.all_data_path[index])
        name = self.name_list[index]
        img = d['image'].astype(np.float32)
        seg = d['label']
        pl = d['pl']

        if self.weak_strong_aug:
            transformed_w = self.augmenter_w(image=img, mask=seg)
            img_w = transformed_w['image']
            seg = transformed_w['mask']
            transformed_s = self.augmenter_s(image=img_w.numpy().transpose((1, 2, 0)))
            img_s = transformed_s['image']
            return img_w, img_s, seg, name
        else:
            transformed = self.augmenter(image=img, mask=pl)
            img = transformed['image']
            img = img.to(torch.float32)
            seg = transformed['mask']
            seg = seg.to(torch.long)
            return img, seg, name

    def __len__(self):
        return len(self.all_data_path)
