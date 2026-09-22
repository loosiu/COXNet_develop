# Same-stage Thermal-Referenced Prototype Calibration (TRPC)
#
# Complete removal of CLFM:
#   RGB P3/P4/P5/P6, Thermal P3/P4/P5/P6
#     -> same-stride TRPC -> original AAM/HOFM
#
# No DWT/IDWT, frequency fusion, legacy DeConv, or interpolation fallback.
# Compare against same_stage_no_trpc.py to isolate prototype calibration.
_base_ = ['./TRPC.py']

model = dict(
    # Match the RGB FPN strides to Thermal: 8, 16, 32, 64.
    neck=dict(start_level=1),
    use_clfm=[],
    use_trpc=True,
    trpc_cfg=dict(same_stage=True),
)

work_dir = 'work_dir/coxmamba/rgbtdroneperson/trpc/TRPC_same_stage'
