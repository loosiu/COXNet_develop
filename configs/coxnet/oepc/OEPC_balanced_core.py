# Balanced object-context OEPC: a complete, same-stage CLFM replacement.
#
# RGB and Thermal P3 are kept in their own coordinate systems. Each modality
# proposes candidates independently. Thermal-origin candidates are expanded to
# a possible RGB bag and RGB-origin candidates read a possible Thermal bag.
# Object-minus-context Thermal descriptors condition a bounded residual on the
# original RGB feature; the original Thermal feature enters AAM unchanged.
#
# Detector counterfactual utility is intentionally disabled in this controlled
# experiment. EDL uncertainty is only a router input and a small auxiliary
# objective; standard foreground heads build the object/context descriptors.
_base_ = ['../coxnet_r50_fpn_1x_rgbtdroneperson.py']

model = dict(
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
        candidate_threshold=0.1,
        # This is a per-modality cap. The union therefore has at most 64
        # centers before overlapping support regions are merged.
        max_candidates=32,
        peak_kernel=3,
        support_kernel=3,
        residual_scale=0.2,
        feature_scale_floor=0.1,
        modulation_init_std=1e-2,
        use_rgb_candidates=True,
        use_edl=True,
        use_detector_utility=False,
        edl_kl_weight=1e-3,
        targetness_loss_weight=0.1,
        foreground_loss_weight=0.1,
        contrastive_loss_weight=0.05,
        edl_loss_weight=0.01,
        utility_loss_weight=0.0,
        trial_detection_loss_weight=0.0,
        utility_penalty_weight=0.0,
        focal_gamma=2.0,
    ),
)

work_dir = (
    'work_dir/coxmamba/rgbtdroneperson/oepc_balanced_core/default')
