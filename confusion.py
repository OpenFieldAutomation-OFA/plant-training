import torch
from mmpretrain.evaluation import ConfusionMatrix
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import tempfile
from mmpretrain.utils import register_all_modules

torch.set_printoptions(threshold=10_000)

config = 'plantclef-pretrained.py'
checkpoint = 'work_dirs/plantclef-pretrained/epoch_9.pth'

cfg = Config.fromfile(config)
register_all_modules(init_default_scope=False)
cfg.test_evaluator = dict(type='ConfusionMatrix')
cfg.load_from = str(checkpoint)
with tempfile.TemporaryDirectory() as tmpdir:
    cfg.work_dir = tmpdir
    runner = Runner.from_cfg(cfg)
    classes = runner.test_loop.dataloader.dataset.metainfo.get(
        'classes')
    test_result = runner.test()

# print(cm)
print("yo")