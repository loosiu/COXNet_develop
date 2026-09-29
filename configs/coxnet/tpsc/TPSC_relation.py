"""Canonical TPSC with prototype-set relation mixing."""
_base_ = ['../coxnet_r50_fpn_1x_rgbtdroneperson.py']

model = dict(
    neck=dict(start_level=1),
    neck_t=dict(start_level=1),
    use_clfm=[],
    use_trpc=False,
    use_oepc=False,
    use_topc=False,
    use_prldfc=False,
    use_tpsc=True,
    wf_loss=True,
    wf_loss_mode='kl_v2',
    wf_loss_weight=0.1,
    tpsc_cfg=dict(
        apply_levels=(0, 1),
        prototype_dim=64,
        num_slots=8,
        num_heads=4,
        relation_depth=1,
        modulation_bound=0.1,
        init_std=1e-2,
        use_relation=True,
        coverage_loss_weight=0.05,
        diversity_loss_weight=0.01))

work_dir = 'work_dir/coxmamba/rgbtdroneperson/tpsc/seed0'
