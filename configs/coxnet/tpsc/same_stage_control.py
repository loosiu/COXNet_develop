"""Same-stage CLFM-free control matched to the TPSC training recipe."""
_base_ = ['../coxnet_r50_fpn_1x_rgbtdroneperson.py']

model = dict(
    neck=dict(start_level=1),
    neck_t=dict(start_level=1),
    use_clfm=[],
    use_trpc=False,
    use_oepc=False,
    use_topc=False,
    use_prldfc=False,
    use_tpsc=False,
    wf_loss=True,
    wf_loss_mode='kl_v2',
    wf_loss_weight=0.1)

work_dir = 'work_dir/coxmamba/rgbtdroneperson/tpsc/control_seed0'
