# Prototype-Routed Local Dynamic Frequency Calibration.
_base_ = ['../coxnet_r50_fpn_1x_rgbtdroneperson.py']

model = dict(
    neck=dict(start_level=1),
    neck_t=dict(start_level=1),
    use_clfm=[],
    use_trpc=False,
    use_oepc=False,
    use_topc=False,
    use_prldfc=True,
    wf_loss=True,
    wf_loss_mode='kl_v2',
    wf_loss_weight=0.1,
    prldfc_cfg=dict(
        apply_levels=(0, 1),
        frequency_dim=64,
        prototype_dim=64,
        num_bands=3,
        search_radius=(2, 2),
        seed_prior=0.01,
        seed_threshold=0.10,
        seed_temperature=0.25,
        frequency_temperature=0.02,
        min_band_width=0.05,
        level_scale_ranges=((0, 32), (16, 64)),
        residual_epsilon=(0.1, 0.1),
        seed_loss_weight=0.10,
        offset_loss_weight=0.05,
        scale_loss_weight=0.02,
        cardinality_loss_weight=0.01))

work_dir = 'work_dir/coxmamba/rgbtdroneperson/prldfc/seed0'
