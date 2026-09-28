_base_ = ['./PRLDFC.py']

model = dict(
    prldfc_cfg=dict(
        apply_levels=(0,),
        search_radius=(2,),
        level_scale_ranges=((0, 32),),
        residual_epsilon=(0.1,)))

work_dir = 'work_dir/coxmamba/rgbtdroneperson/prldfc_p3/seed0'
