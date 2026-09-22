# Same-stage Thermal-Conditioned Prototype Calibration (TRPC)
#
# Complete removal of CLFM:
#   RGB P3/P4/P5/P6, Thermal P3/P4/P5/P6
#     -> Thermal scene prototypes
#     -> per-RGB-location soft attention
#     -> spatial FiLM scale/shift of RGB -> original AAM/HOFM
#
# No DWT/IDWT, frequency fusion, legacy DeConv, or interpolation fallback.
# Compare against same_stage_no_trpc.py to isolate prototype calibration.
_base_ = ['../coxnet_r50_fpn_1x_rgbtdroneperson.py']

model = dict(
    # Match the RGB FPN strides to Thermal: 8, 16, 32, 64.
    neck=dict(start_level=1),
    use_clfm=[],
    use_trpc=True,
    trpc_cfg=dict(
        variant='thermal_conditioning',
        num_prototypes=8,
        embed_dim=64,
        objectness_prior=0.1,
        objectness_bias=1.0,
        epsilon=0.1,
        modulation_init_std=1e-3,
        # Thermal targetness remains as weak object-aware pooling supervision.
        # No matching, correspondence, or alignment loss is used.
        targetness_loss_weight=0.1,
        diversity_loss_weight=0.0,
        focal_gamma=2.0,
    ),
)

work_dir = 'work_dir/coxmamba/rgbtdroneperson/trpc/TRPC_same_stage'
