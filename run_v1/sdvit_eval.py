"""Evaluate an SD-ViT checkpoint at every exit under the three protocols of the paper.

  --protocol self           probe and gallery both embedded at exit k            (Tab. 1, Fig. 2a-c)
  --protocol gallery_fixed  exit-k probe against a gallery enrolled at full depth (Tab. 5, Fig. 2e-g)
  --protocol threshold      one threshold calibrated on full-depth development
                            scores, applied unchanged at every exit               (Tab. 6)

Example:
  python sdvit_eval.py --ckpt_dir ../../../../pretrained_models/sdvit/sdvit_vit_s12_wf4m --protocol self
"""
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, indicator=["__root__.txt"], pythonpath=True, dotenv=True)

import argparse
import os
import sys
from functools import partial

import numpy as np

np.bool = np.bool_  # mxnet 1.9.1, imported by CVLface's verification code, still uses these aliases
np.object = np.object_

import pandas as pd
import sklearn.preprocessing
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger

from evaluations import get_evaluator_by_name
from evaluations.ijbbc.evaluate import image2template_feature, verification
from fabric.fabric import setup_dataloader_from_dataset
from general_utils.config_utils import load_config
from models import get_model
from models.base.utils import load_state_dict_from_path
from pipelines import pipeline_from_name
from sdvit_gallery_fixed import GALLERY_FIXED_EVALUATORS

HQ_SETS = ['lfw', 'agedb_30', 'cfpfp', 'cplfw', 'calfw']
IJBC_KEY = 'IJBC_gt_aligned/Norm:True_Det:True_tpr_at_fpr_{}'  # the IJB-C setting reported in the paper


def parse_exits(exits, depth):
    if exits.strip().lower() == 'all':
        return list(range(1, depth + 1))
    return [int(k) for k in exits.split(',') if k.strip()]


def build_evaluators(eval_config, protocol, pipeline, fabric):
    evaluators = []
    for name, info in eval_config.per_epoch_evaluations.items():
        path = os.path.join(eval_config.data_root, info.path)
        if protocol == 'self':
            evaluator = get_evaluator_by_name(info.evaluation_type, name, path, pipeline.make_test_transform(),
                                              fabric, info.batch_size, info.num_workers)
        else:
            if protocol == 'threshold' and info.evaluation_type not in ('verification', 'ijbbc'):
                continue  # identification benchmarks have no verification pairs to threshold
            assert os.path.isdir(path), f'evaluation data not found at {path}, check DATA_ROOT in cvlface/.env'
            evaluator = GALLERY_FIXED_EVALUATORS[info.evaluation_type](
                name, path, pipeline.make_test_transform(), fabric, info.batch_size, info.num_workers)
        evaluator.integrity_check(info.color_space, pipeline.color_space)
        evaluator.log = lambda *a, **kw: None
        evaluators.append(evaluator)
    return evaluators


def paper_summary(results):
    """Per-exit table with the metrics reported in the paper, in percent."""
    rows = []
    for exit_index, result in sorted(results.items()):
        row = {'exit': exit_index}
        hq = [result.get(f'{name}/acc') for name in HQ_SETS]
        row.update({name: acc for name, acc in zip(HQ_SETS, hq) if acc is not None})
        if all(acc is not None for acc in hq):
            row['HQ(5)'] = float(np.mean(hq))
        for key, column in (('tinyface/rank-1', 'TinyFace R-1'), ('tinyface/rank-5', 'TinyFace R-5')):
            if key in result:
                row[column] = result[key]
        for far in ('1e-05', '0.0001', '0.001'):
            if IJBC_KEY.format(far) in result:
                row[f'IJB-C TAR@{float(far):.0e}'] = result[IJBC_KEY.format(far)]
        rows.append(row)
    return pd.DataFrame(rows)


def pair_scores(evaluator, collection, collection_flip):
    """Cosine score and binary label for every verification pair of the benchmark."""
    embeddings = (collection['features'] + collection_flip['features']).numpy()
    if hasattr(evaluator, 'meta'):  # IJB-B/C template protocol, Norm:True_Det:True
        meta = evaluator.meta
        feats = embeddings * meta['faceness_scores'][:, np.newaxis]
        template_feats, unique_templates = image2template_feature(feats, meta['templates'], meta['medias'],
                                                                  dummy=False)
        score = verification(template_feats, unique_templates, meta['p1'], meta['p2'])
        return score.astype(np.float64), np.asarray(meta['label']).astype(np.int64)
    embeddings = sklearn.preprocessing.normalize(embeddings)  # pair protocol of the LFW family
    score = np.sum(embeddings[0::2] * embeddings[1::2], axis=1)
    return score.astype(np.float64), collection['is_same'].numpy()[::2].astype(np.int64)


def threshold_at_far(scores, labels, target_far):
    """Smallest threshold whose impostor acceptance rate does not exceed target_far."""
    impostor = np.sort(scores[labels == 0])[::-1]
    k = int(np.floor(target_far * len(impostor)))
    if k <= 0:
        return float(impostor[0]) + 1e-12
    return float(impostor[k - 1])


def far_tar(scores, labels, tau):
    return float((scores[labels == 0] >= tau).mean()), float((scores[labels == 1] >= tau).mean())


def calibrate(scores_per_exit, label, depth, args):
    """One tau on the development half of the full-depth scores, applied to the test half at every exit."""
    is_dev = np.random.RandomState(args.split_seed).rand(len(label)) < args.dev_fraction
    tau = threshold_at_far(scores_per_exit[depth][is_dev], label[is_dev], args.target_far)
    rows = []
    for exit_index, score in sorted(scores_per_exit.items()):
        far, tar = far_tar(score[~is_dev], label[~is_dev], tau)
        tau_k = threshold_at_far(score[is_dev], label[is_dev], args.target_far)  # per-exit re-calibration
        _, tar_k = far_tar(score[~is_dev], label[~is_dev], tau_k)
        rows.append({'exit': exit_index, 'tau_full_depth': tau, 'FAR_fixed_tau': far, 'TAR_fixed_tau': 100 * tar,
                     'tau_per_exit': tau_k, 'TAR_per_exit_tau': 100 * tar_k, 'TAR_loss': 100 * (tar_k - tar),
                     'genuine_mean': score[label == 1].mean(), 'impostor_mean': score[label == 0].mean()})
    return pd.DataFrame(rows), is_dev, tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, help='directory with model.pt and model.yaml')
    parser.add_argument('--protocol', type=str, default='self', choices=['self', 'gallery_fixed', 'threshold'])
    parser.add_argument('--exits', type=str, default='all', help='"all" or comma-separated 1-based exits, e.g. 3,6,12')
    parser.add_argument('--eval_config_name', type=str, default='sdvit')
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32-true')
    parser.add_argument('--target_far', type=float, default=1e-4, help='threshold protocol: calibration FAR')
    parser.add_argument('--dev_fraction', type=float, default=0.5, help='threshold protocol: development split')
    parser.add_argument('--split_seed', type=int, default=2048, help='threshold protocol: split seed')
    parser.add_argument('--output_dir', type=str, default='')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.ckpt_dir, 'sdvit_eval', args.protocol)
    os.makedirs(output_dir, exist_ok=True)
    eval_config = load_config(f'evaluations/configs/{args.eval_config_name}.yaml')

    model_config = load_config(os.path.join(args.ckpt_dir, 'model.yaml'))
    model_config.start_from = ''
    if '/vit/' in model_config.yaml_path:
        # A plain CVLface ViT (e.g. the Baseline) has the same parameters; reading its intermediate
        # blocks through its own head is the Baseline / training-free ViT-FREE read-out of the paper.
        model_config.yaml_path = model_config.yaml_path.replace('/vit/', '/vit_SD/')
    sdvit = get_model(model_config)
    assert hasattr(sdvit, 'set_exit'), f'{model_config.yaml_path} is not an SD-ViT model (models/vit_SD)'
    sdvit.load_state_dict(load_state_dict_from_path(os.path.join(args.ckpt_dir, 'model.pt')), strict=True)
    depth = sdvit.depth
    exits = parse_exits(args.exits, depth)

    fabric = Fabric(precision=args.precision, accelerator='auto', strategy='ddp', devices=args.num_gpu,
                    loggers=[CSVLogger(root_dir=output_dir, flush_logs_every_n_steps=1)])
    fabric.launch()
    fabric.setup_dataloader_from_dataset = partial(setup_dataloader_from_dataset, fabric=fabric, seed=2048)
    model = fabric.setup(sdvit)
    pipeline = pipeline_from_name('infer_model_pipeline', model)
    pipeline.integrity_check(dataset_color_space='RGB')
    evaluators = build_evaluators(eval_config, args.protocol, pipeline, fabric)

    if args.protocol != 'self':
        if fabric.local_rank == 0:
            print(f'Enrolling the gallery once at full depth (exit {depth})')
        sdvit.set_exit(None)
        for evaluator in evaluators:
            evaluator.cache_gallery(pipeline)

    if args.protocol in ('self', 'gallery_fixed'):
        results = {}
        for exit_index in exits:
            sdvit.set_exit(exit_index)
            if fabric.local_rank == 0:
                print(f'Exit {exit_index}/{depth}')
            result = {}
            for evaluator in evaluators:
                result.update({f'{evaluator.name}/{k}': v for k, v in evaluator.evaluate(pipeline).items()})
            results[exit_index] = result
        sdvit.set_exit(None)
        if fabric.local_rank != 0:
            sys.exit(0)
        pd.DataFrame.from_dict(results, orient='index').rename_axis('exit').to_csv(
            os.path.join(output_dir, f'results_{args.protocol}.csv'))
        summary = paper_summary(results)
        summary.to_csv(os.path.join(output_dir, f'summary_{args.protocol}.csv'), index=False)
        print(f'\n{args.protocol} protocol, {args.ckpt_dir}\n' + summary.round(2).to_string(index=False))
        print(f'\nwritten to {output_dir}')
        return

    # threshold protocol: pair scores at every exit against the fixed gallery
    if depth not in exits:
        exits.append(depth)  # full depth is the calibration reference
    scores = {evaluator.name: {} for evaluator in evaluators}
    labels = {}
    for exit_index in exits:
        sdvit.set_exit(exit_index)
        if fabric.local_rank == 0:
            print(f'Scoring probes at exit {exit_index}/{depth}')
        for evaluator in evaluators:
            probe = evaluator.extract(pipeline, flip_images=False)
            probe_flip = evaluator.extract(pipeline, flip_images=True)
            if fabric.local_rank == 0:
                scores[evaluator.name][exit_index], labels[evaluator.name] = pair_scores(
                    evaluator, *evaluator.mix(probe, probe_flip))
    sdvit.set_exit(None)
    if fabric.local_rank != 0:
        sys.exit(0)

    for name, per_exit in scores.items():
        table, is_dev, tau = calibrate(per_exit, labels[name], depth, args)
        table.to_csv(os.path.join(output_dir, f'threshold_{name}.csv'), index=False)
        np.savez_compressed(os.path.join(output_dir, f'scores_{name}.npz'), label=labels[name], is_dev=is_dev,
                            tau=tau, target_far=args.target_far, **{f'exit_{k}': v for k, v in per_exit.items()})
        print(f'\n{name}: tau = {tau:.4f} calibrated at full depth for FAR = {args.target_far:g}')
        print(table[['exit', 'FAR_fixed_tau', 'TAR_fixed_tau', 'TAR_per_exit_tau', 'TAR_loss']]
              .to_string(index=False, float_format=lambda v: f'{v:.6g}'))
    print(f'\nwritten to {output_dir}')


if __name__ == '__main__':
    main()
