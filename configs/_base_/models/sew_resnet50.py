# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SEWResNet',
        block_type='SEWBasicBlock',
        layers=[3, 4, 6, 3],
        width=[64, 128, 256, 512],
        stride=[1, 2, 2, 2],
        num_classes=1000,
        in_channels=3,
        zero_init_residual=True,
        groups=1, width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        cnf_list=['add'],
        neuron_cfg=dict(
            type='IFNode',
            surrogate_function=dict(
                type='ATan'
            )
        ),
    ),
    head=dict(
        type='ClsHead',
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
