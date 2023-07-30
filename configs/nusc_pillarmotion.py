import itertools
import logging


from motion.utils.config_tool import get_downsample_factor

norm_cfg = None


_pc_range = [-32.0, -32.0, -5.0, 32.0, 32.0, 3.0]
_voxel_size = [0.25, 0.25, 8] 

voxel_generator = dict(
    range=_pc_range,
    voxel_size=_voxel_size,
    nsweeps=5,
    max_points_in_voxel=20,
    max_voxel_num=30000,
)

# model settings
model = dict(
    type="MotionNet",
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[32],
        num_input_features=5,
        with_distance=False,
        voxel_size=_voxel_size,
        pc_range=_pc_range,
        norm_cfg=norm_cfg,
    ),
    backbone=dict(type="PointPillarsScatter", ds_factor=1, norm_cfg=norm_cfg,),
    neck=dict(
        type="STPN",
        height_feat_size=32, 
    ),
    head=dict(
        type="MotionHead",
        in_channels=32,  
        channels=32,
        
    ),
    voxel_cfg=voxel_generator,
)


# dataset settings
dataset_type = "NuScenesDataset"
data_root = "data/v1.0-trainval"


train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,

)



cam_name = ['front', 'front_left', 'front_right', 'back', 'back_left', 'back_right']

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, pc_range=_pc_range[3], nsweeps=5, mode='train'),
    dict(type="Preprocess", cfg=train_preprocessor, pc_range=_pc_range, voxel_size=_voxel_size, cam_name=cam_name),
    dict(type="Voxelization", cfg=voxel_generator, ),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, pc_range=_pc_range[3], nsweeps=5, mode='val'),
    dict(type="Preprocess", cfg=val_preprocessor, pc_range=_pc_range, voxel_size=_voxel_size,cam_name=cam_name),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="Reformat"),
]

train_anno = "data/v1.0-trainval/infos_train_withcam.pkl"
val_anno = "data/v1.0-trainval/infos_val_withcam.pkl"
test_anno = "data/v1.0-trainval/infos_test_withcam.pkl"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        test_mode=False,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        pipeline=test_pipeline,
    ),
)

# optimizer
optimizer = dict(
    type="AdamW", amsgrad=0.0, weight_decay=0.0001, lr=0.0001
)

"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy in training hooks
lr_config = dict(
    type="exponential_decay", initial_learning_rate=0.0001, decay_length=0.125, decay_factor=0.9, staircase=True,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
# yapf:enable
# runtime settings
total_epochs = 200
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = "./experiments/motionnet"
load_from = None
resume_from = None
#workflow = [("train", 1), ("val", 1)]
workflow = [("train", 1)]
