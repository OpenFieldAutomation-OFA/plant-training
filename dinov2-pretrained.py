_base_ = 'plantclef-pretrained.py'

model = dict(
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/mnt/data/plantclef/pretrained/dinov2.pth'
    )
)