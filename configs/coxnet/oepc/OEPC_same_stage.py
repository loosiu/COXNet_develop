# Object-centric Evidential Prototype Calibration (OEPC)
#
# Complete CLFM replacement:
#   RGB P3/P4/P5/P6, Thermal P3/P4/P5/P6 (same strides)
#     -> Thermal-coordinate candidate selection
#     -> foreground-weighted object/context prototype difference
#     -> distance-aware soft RGB candidate bag
#     -> EDL-informed, detector-utility-supervised routing
#     -> support- and norm-bounded RGB-only residual calibration
#     -> original AAM/HOFM(RGB_calibrated, Thermal_original)
#
# No cross-stage pairing, DeConv, DWT/IDWT, frequency fusion, or global scene
# prototype is instantiated. The first controlled experiment applies OEPC only
# at P3, where tiny objects retain meaningful spatial support; other levels
# pass same-stage RGB and Thermal directly to the original AAM/HOFM.
_base_ = ['../coxnet_r50_fpn_1x_rgbtdroneperson.py']

model = dict(
    # Both FPNs now output strides 8, 16, 32, and 64.
    neck=dict(start_level=1),
    neck_t=dict(start_level=1),
    use_clfm=[],
    use_trpc=False,
    use_oepc=True,
    oepc_cfg=dict(
        apply_levels=(0,),
        embed_dim=64,
        contrast_dim=32,
        object_kernel=3,
        context_kernel=7,
        search_radius=2,
        search_temperature=0.2,
        distance_prior_weight=0.1,
        candidate_prior=0.1,
        candidate_threshold=0.05,
        max_candidates=100,
        peak_kernel=3,
        support_kernel=3,
        residual_scale=0.2,
        feature_scale_floor=0.1,
        modulation_init_std=1e-2,
        edl_kl_weight=1e-3,
        # Counterfactual loss differences are typically 1e-4--1e-3 early in
        # training; this temperature keeps the detached utility target useful.
        utility_temperature=0.01,
        utility_penalty_weight=0.01,
        trial_detection_loss_weight=0.05,
        targetness_loss_weight=0.1,
        contrastive_loss_weight=0.05,
        edl_loss_weight=0.01,
        utility_loss_weight=0.05,
        focal_gamma=2.0,
    ),
)

work_dir = 'work_dir/coxmamba/rgbtdroneperson/oepc/OEPC_same_stage'
