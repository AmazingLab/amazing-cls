_base_ = [
    '../_base_/datasets/dvs_gesture_t16.py',
    '../_base_/default_runtime.py'
]

# 88.2500%
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='org4dvs_gesture'
    ),
    head=dict(
        type='ClsHead',
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-5)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineRestart',
    periods=[30, 60, 150],
    restart_weights=[1., 0.9, 0.6],
    min_lr_ratio=0.001,
    warmup='linear',
    warmup_iters=500,
)
runner = dict(type='EpochBasedRunner', max_epochs=90)
fp16 = None

