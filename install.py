"""Add SD-ViT to a CVLface checkout (https://github.com/mk-minchul/CVLface).

    python install.py --cvlface /path/to/CVLface

Copies the files under run_v1/ into CVLface/cvlface/research/recognition/code/run_v1 and registers
the SD-ViT model, its training pipeline and the gridsample_v2 augmenter in CVLface's registries.
Safe to re-run: files and registrations that are already in place are skipped.
"""
import argparse
import filecmp
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (registry file, line to anchor on, text that replaces it)
REGISTRATIONS = [
    ('models/__init__.py',
     "    if '/vit/' in model_config.yaml_path:\n",
     "    if '/vit_SD/' in model_config.yaml_path:\n"
     "        from .vit_SD import load_model as load_vit_SD_model\n"
     "        model = load_vit_SD_model(model_config)\n"
     "        print('Loaded SD-ViT model')\n"
     "    elif '/vit/' in model_config.yaml_path:\n"),
    ('pipelines/__init__.py',
     "from .train_model_cls_pipeline import TrainModelClsPipeline\n",
     "from .train_model_cls_pipeline import TrainModelClsPipeline\n"
     "from .train_model_sd_cls_pipeline import TrainModelSDClsPipeline\n"),
    ('pipelines/__init__.py',
     "    if pipeline_config.name == 'TrainModelClsPipeline':\n",
     "    if pipeline_config.name == 'TrainModelSDClsPipeline':\n"
     "       pipeline = TrainModelSDClsPipeline(model, classifier, optimizer, lr_scheduler,\n"
     "                                          loss_weight=pipeline_config.get('loss_weight', 0.7))\n"
     "    elif pipeline_config.name == 'TrainModelClsPipeline':\n"),
    ('data_augs/__init__.py',
     "from .gridsample_augmenter import GridSampleAugmenter\n",
     "from .gridsample_augmenter import GridSampleAugmenter\n"
     "from .gridsample_augmenter_v2 import GridSampleAugmenterV2\n"),
    ('data_augs/__init__.py',
     "    elif augmentation_version == 'gridsample':\n",
     "    elif augmentation_version == 'gridsample_v2':\n"
     "        augmenter = GridSampleAugmenterV2(aug_params, input_size=112)\n"
     "    elif augmentation_version == 'gridsample':\n"),
]


def copy_files(src_root, dst_root, force):
    for dirpath, _, filenames in os.walk(src_root):
        for filename in filenames:
            src = os.path.join(dirpath, filename)
            rel = os.path.relpath(src, src_root)
            dst = os.path.join(dst_root, rel)
            if os.path.exists(dst):
                if filecmp.cmp(src, dst, shallow=False):
                    continue
                if not force:
                    sys.exit(f'{dst} exists and differs from the SD-ViT version; rerun with --force to overwrite')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f'copied    {rel}')


def register(dst_root):
    for rel, anchor, replacement in REGISTRATIONS:
        path = os.path.join(dst_root, rel)
        with open(path) as f:
            text = f.read()
        if replacement in text:
            continue
        if text.count(anchor) != 1:
            sys.exit(f'could not find a unique anchor in {rel}; register SD-ViT by hand:\n{replacement}')
        with open(path, 'w') as f:
            f.write(text.replace(anchor, replacement))
        print(f'registered in {rel}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvlface', required=True, help='root of the CVLface repository')
    parser.add_argument('--force', action='store_true', help='overwrite files that differ')
    args = parser.parse_args()

    dst_root = os.path.join(args.cvlface, 'cvlface', 'research', 'recognition', 'code', 'run_v1')
    if not os.path.isfile(os.path.join(dst_root, 'train.py')):
        sys.exit(f'{dst_root} does not look like CVLface run_v1 (train.py not found)')
    copy_files(os.path.join(HERE, 'run_v1'), dst_root, args.force)
    register(dst_root)
    print(f'SD-ViT installed into {dst_root}')


if __name__ == '__main__':
    main()
