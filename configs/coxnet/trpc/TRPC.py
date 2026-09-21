# Thermal-Referenced Prototype Calibration (TRPC)
#
# Complete CLFM replacement:
#   RGB/T -> object-aware prototypes -> mutual semantic matching
#         -> Thermal-guided RGB prototype residual
#         -> RGB-attention reconstruction -> original AAM/HOFM
#
# The legacy CLFM DeConv is retained only as the cross-level resolution matcher;
# DWT, LL fusion, HF gating and IDWT are not used.
# Thermal prototypes are stop-gradient references on the calibration path, and
# the final RGB residual projection is zero-initialized (iteration-zero identity).
_base_ = ['../coxnet_r50_fpn_1x_rgbtdroneperson.py']

model = dict(
    use_clfm=[],
    use_trpc=True,
    trpc_cfg=dict(
        num_prototypes=8,
        embed_dim=64,
        match_temperature=0.2,
        match_topk=1,
        mutual_matching=True,
        objectness_prior=0.1,
        objectness_bias=1.0,
        residual_scale=0.2,
        learn_residual_scale=True,
        # Only the thermal objectness head uses the thermal-coordinate GT.
        # RGB objectness is learned through the detector objective because a
        # thermal box is not a valid RGB-coordinate target under misalignment.
        targetness_loss_weight=0.1,
        diversity_loss_weight=0.01,
        focal_gamma=2.0,
    ),
)

work_dir = 'work_dir/coxmamba/rgbtdroneperson/trpc/TRPC'
