"""Gallery-fixed versions of CVLface's evaluators (Sec. 4.4 of the paper).

The enrolled (gallery) side of every benchmark is embedded once with the full-depth
model and cached by `cache_gallery`. Every later `evaluate` call embeds only the
probe side with whatever exit the model is set to, and scores probes against that
fixed gallery:

  verification sets (LFW, AgeDB-30, CFP-FP, CPLFW, CALFW): the second image of
      every pair (odd index) is the gallery, the first image is the probe;
  TinyFace: gallery and distractor images are the gallery, probe images the probe;
  IJB-B/C: every image of a template listed in p1 is the gallery, the rest probes.
"""
import os

import numpy as np
import torch

from evaluations.ijbbc_evaluator import IJBBCEvaluator
from evaluations.tinyface_evaluator import TinyFaceEvaluator
from evaluations.verification_evaluator import VerificationEvaluator


class GalleryFixedMixin:

    def gallery_mask(self, collection):
        raise NotImplementedError

    @torch.no_grad()
    def cache_gallery(self, full_depth_pipeline):
        full_depth_pipeline.eval()
        self._gallery = self.extract(full_depth_pipeline, flip_images=False)
        self._gallery_flip = self.extract(full_depth_pipeline, flip_images=True)
        if self.fabric.local_rank == 0:
            self._gallery_mask = torch.from_numpy(self.gallery_mask(self._gallery))

    def mix(self, probe, probe_flip):
        """Probe collections with their gallery entries replaced by the cached full-depth ones."""
        mixed = []
        for collection, gallery in ((probe, self._gallery), (probe_flip, self._gallery_flip)):
            features = collection['features'].clone()
            features[self._gallery_mask] = gallery['features'][self._gallery_mask]
            mixed.append({**collection, 'features': features})
        return mixed

    @torch.no_grad()
    def evaluate(self, pipeline, epoch=0, step=0, n_images_seen=0):
        assert hasattr(self, '_gallery'), 'call cache_gallery() with the full-depth model first'
        pipeline.eval()
        probe = self.extract(pipeline, flip_images=False)
        probe_flip = self.extract(pipeline, flip_images=True)
        if self.fabric.local_rank != 0:
            return {}
        return self.compute_metric(*self.mix(probe, probe_flip))


class GalleryFixedVerificationEvaluator(GalleryFixedMixin, VerificationEvaluator):

    def gallery_mask(self, collection):
        mask = np.zeros(len(collection['features']), dtype=bool)
        mask[1::2] = True
        return mask


class GalleryFixedTinyFaceEvaluator(GalleryFixedMixin, TinyFaceEvaluator):

    def gallery_mask(self, collection):
        key = lambda path: os.path.splitext(os.path.basename(str(path)))[0]
        index_of = {key(p): i for i, p in enumerate(collection['image_paths'])}
        mask = np.zeros(len(collection['image_paths']), dtype=bool)
        for path in list(self.meta['gallery_paths']) + list(self.meta['distractor_paths']):
            if key(path) in index_of:
                mask[index_of[key(path)]] = True
        return mask


class GalleryFixedIJBBCEvaluator(GalleryFixedMixin, IJBBCEvaluator):

    def gallery_mask(self, collection):
        gallery_templates = set(self.meta['p1'].tolist())
        return np.array([t in gallery_templates for t in np.asarray(self.meta['templates'])], dtype=bool)


GALLERY_FIXED_EVALUATORS = {
    'verification': GalleryFixedVerificationEvaluator,
    'tinyface': GalleryFixedTinyFaceEvaluator,
    'ijbbc': GalleryFixedIJBBCEvaluator,
}
