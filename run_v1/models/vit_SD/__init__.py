from torchvision import transforms

from ..base import BaseModel
from .vit_SD import VisionTransformer

# Same backbones as CVLface's ViT (models/vit), d=512 and 8x8 patches on 112x112 faces.
ARCHS = {
    'small': dict(depth=12, num_heads=8, mlp_ratio=5),      # ViT-S/12
    'small_15': dict(depth=15, num_heads=8, mlp_ratio=5),   # ViT-S/15
    'base': dict(depth=24, num_heads=16, mlp_ratio=3),      # ViT-B/24
}


class ViTModel(BaseModel):

    def __init__(self, net, config):
        super(ViTModel, self).__init__(config)
        self.net = net

    @classmethod
    def from_config(cls, config):
        if config.name not in ARCHS:
            raise NotImplementedError(f'unknown SD-ViT architecture {config.name}')
        net = VisionTransformer(img_size=112, patch_size=8, num_classes=config.output_dim, embed_dim=512,
                                drop_path_rate=0.1, **ARCHS[config.name])
        model = cls(net, config)
        model.eval()
        return model

    @property
    def depth(self):
        return self.net.depth

    def set_exit(self, exit_index):
        """Run only the first `exit_index` blocks at inference (None = full depth)."""
        self.net.exit_index = exit_index

    def forward(self, x):
        if self.input_color_flip:
            x = x.flip(1)
        return self.net(x)

    def make_train_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def make_test_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


def load_model(model_config):
    return ViTModel.from_config(model_config)
