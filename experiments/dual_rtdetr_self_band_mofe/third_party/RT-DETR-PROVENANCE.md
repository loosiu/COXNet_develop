# Vendored RT-DETR provenance

Source: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR), commit `29320b6fd828f8e0987a71426cf2d961b09dfed7`, `rtdetr_pytorch/src` tree. License: Apache-2.0, copied verbatim to [`RT-DETR-LICENSE`](RT-DETR-LICENSE). Original source notices and copyright text are retained.

The following files were copied into `dyfclt/vendor/`:

| Vendored file | Original path under `rtdetr_pytorch/src/` | Changes |
| --- | --- | --- |
| `common.py` | `nn/backbone/common.py` | None |
| `presnet.py` | `nn/backbone/presnet.py` | Removed `src.core` registry import and decorator; pretrained loading in `DyFCLT` is explicit. |
| `hybrid_encoder.py` | `zoo/rtdetr/hybrid_encoder.py` | Removed `src.core` registry import and decorator. |
| `rtdetr_decoder.py` | `zoo/rtdetr/rtdetr_decoder.py` | Removed `src.core` registry import and decorator; changed deprecated `F.sigmoid` to `torch.sigmoid`; added eval-only `forced_indices` and `return_query_metadata` diagnostic options. Ordinary forward calls leave both options disabled. |
| `denoising.py` | `zoo/rtdetr/denoising.py` | None |
| `utils.py` | `zoo/rtdetr/utils.py` | None |
| `box_ops.py` | `zoo/rtdetr/box_ops.py` | None |
| `matcher.py` | `zoo/rtdetr/matcher.py` | Removed `src.core` registry import and decorator; changed deprecated `F.sigmoid` to `torch.sigmoid`. |
| `rtdetr_criterion.py` | `zoo/rtdetr/rtdetr_criterion.py` | Replaced `src.misc.dist` import with local shim; removed `src.core` registry import and decorator; changed deprecated `F.sigmoid` to `torch.sigmoid`. |

`distributed.py` and `__init__.py` are local glue, not copies of upstream. The shim calls `torch.distributed` for initialized process groups and returns one process otherwise. No `src` path insertion or upstream package install is used. The official detection decoder, Hungarian matching, VFL/L1/GIoU criterion, auxiliary losses, and denoising logic remain in the vendored files. The paper-specific encoder and foreground mask loss live in `dyfclt/model.py` and `dyfclt/losses.py`.

The optional `dyfclt/training.py` recipe (2026-10-01) follows the same commit's `configs/rtdetr/include/optimizer.yml` name-based optimizer groups and `src/optim/ema.py` ModelEMA floating-state update/decay warmup (by lyuwenyu; source references the timm/YOLOv5 EMA implementations). Local adaptations support the two backbone names, configurable base-LR scaling, explicit disabled legacy defaults, and persistence of decay/warmups/updates. See the retained Apache-2.0 license.
