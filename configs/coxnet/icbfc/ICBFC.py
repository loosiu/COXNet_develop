# Instance-Conditioned Cross-Band Frequency Calibration.
_base_ = ['../coxnet_r50_fpn_1x_rgbtdroneperson.py']

model = dict(
    use_clfm=[],
    use_trpc=False,
    use_oepc=False,
    use_topc=False,
    use_prldfc=False,
    use_tpsc=False,
    use_icbfc=True,
    icbfc_cfg=dict(
        prior_in_channels=256,
        prior_hidden_channels=64,
        token_dim=64,
        score_threshold=0.1,
        candidate_chunk_size=256,
        center_prior=0.01,
        residual_scale=0.1,
        support_cutoff=1e-4,
        aux_loss_weight=0.1))

work_dir = 'work_dir/coxmamba/rgbtdroneperson/icbfc'
