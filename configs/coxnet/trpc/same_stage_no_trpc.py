# Same-stage CLFM-free control for TRPC_same_stage.py.
#
# RGB and Thermal use the same strides (8, 16, 32, 64), then enter the original
# AAM/HOFM directly.  This config changes no assignment, head, loss, or training
# setting.  Comparing it with TRPC_same_stage.py isolates the contribution of
# prototype calibration from both CLFM/DeConv removal and the RGB FPN change.
_base_ = ['../coxnet_r50_fpn_1x_rgbtdroneperson.py']

model = dict(
    neck=dict(start_level=1),
    use_clfm=[],
    use_trpc=False,
)

work_dir = 'work_dir/coxmamba/rgbtdroneperson/trpc/same_stage_no_trpc'
