# Right Thigh XCAT Export Request

Use `right_thigh_muscle_vessel_export.par` to generate the XCAT data for the
right thigh workflow. This file is generated from `general.samp.par`, so it
uses the flat `key = value` format XCAT expects.

The most important outputs to give back are:

1. the NURBS/NRB output created by `nurbs_save = 1`
2. the activity/color-code phantom created by `act_phan_each = 1`
3. the XCAT run log/header with array size, voxel size, and output names
4. any object-label lookup table XCAT emits for `color_code = 1`
5. an optional vessel centerline CSV if your XCAT/Rhino pipeline can export it

The simulation needs the muscle mask as the growth domain, and it needs the
right-leg artery/vein geometry so the femoral trunk and major veins come from
XCAT instead of being guessed by the growth model.

Important settings in the `.par`:

```text
color_code = 1
activ_output_format = 1
nurbs_save = 1
vessel_flag = 1
coronary_art_flag = 0
coronary_vein_flag = 0
pixel_width = 0.10 cm
slice_width = 0.10 cm
x_array_size = 300
y_array_size = 320
startslice = 590
endslice = 990
X_tr = 105.0 mm
Y_tr = -20.0 mm
```

This is intended to output a 1 mm voxel phantom with about `300 x 320 x 401`
voxels. The slice range is an estimate for knee to below hip based on the
previous `vmale50` thigh NRB. `X_tr` and `Y_tr` shift the phantom so the right
thigh is centered in the image field.

If the right thigh appears shifted out of the image, rerun with a wider field
of view first:

```text
x_array_size = 560
y_array_size = 400
X_tr = 0.0
Y_tr = 0.0
```

That wider run may include both legs, but we can crop/filter the right thigh
locally after you send the output.

The right thigh muscle IDs we previously saw in `thigh_xcat_1.nrb` were:

```text
musc160, musc161, musc162, musc164, musc1663,
musc158, musc165, musc159, musc141, musc230,
musc167, musc166, musc79, musc78, musc80,
musc30, musc29, musc27, musc26, musc40,
musc34, musc32, musc28, musc33, musc35
```

The vessel groups to preserve are `arteries_rleg` and `veins_rleg`.
