# Thermal-Anchored Object Prototype Calibration (TOPC).
#
# TOPC completely replaces CLFM. RGB and Thermal use the same FPN stages;
# Thermal P3 candidates provide object-conditioned semantic discrepancies that
# calibrate RGB P3 before the original AAM/HOFM. Thermal features are unchanged.
_base_ = ['../coxnet_r50_fpn_1x_rgbtdroneperson.py']

model = dict(
    neck=dict(start_level=1),
    neck_t=dict(start_level=1),
    use_clfm=[],
    use_trpc=False,
    use_oepc=False,
    use_topc=True,
    wf_loss=False,
    topc_cfg=dict(
        apply_levels=(0,),
        calibration_dim=64,
        object_kernel=3,
        search_radius=2,
        search_temperature=0.2,
        candidate_prior=0.01,
        candidate_threshold=0.05,
        max_candidates=100,
        residual_init_std=1e-2,
        objectness_loss_weight=0.1,
        focal_gamma=2.0,
    ),
)

work_dir = 'work_dir/coxmamba/rgbtdroneperson/topc/seed0'
