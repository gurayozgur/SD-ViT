# Extends GridSampleAugmenter of CVLface (https://github.com/mk-minchul/CVLface),
# MIT License, Copyright (c) 2022 Minchul Kim.
import numpy as np

from data_augs.aug_utils import transform_cv2, transform_torch
from data_augs.gridsample_augmenter import GridSampleAugmenter


class GridSampleAugmenterV2(GridSampleAugmenter):
    """CVLface's GridSampleAugmenter, applied to a sample with probability `aug_prob`.

    Samples that are not augmented still receive the horizontal flip (`hflip_prob`).
    Returns (image, theta) exactly like GridSampleAugmenter.
    """

    def __init__(self, aug_params, input_size=112):
        super().__init__(aug_params, input_size=input_size)
        self.aug_prob = aug_params.get('aug_prob', 1.0)

    def augment(self, sample):
        if np.random.random() < self.aug_prob:
            return super().augment(sample)

        image_np = np.array(sample)
        params = transform_torch.sample_param(scale_min=1.0, scale_max=1.0, rot_prob=0.0, max_rot=0,
                                              hflip_prob=self.aug_params['hflip_prob'], extra_offset=0.0)
        mat = transform_cv2.generate_transform_cv2(image_np, self.input_size, self.input_size, **params)
        aug_sample = transform_cv2.augment_cv2_deterministic(image_np, mat, self.input_size, self.input_size)
        align_input_theta = transform_torch.generate_transform_torch(image_np, self.input_size, self.input_size,
                                                                     **params).squeeze(0)
        return aug_sample, align_input_theta
