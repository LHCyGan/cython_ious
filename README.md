# cython_ious
Cython's standalone iou

## install
```
pip install cython_ious
```

## use
```
from cython_ious import bbox_giou as giou

overlaps = giou(
        np.ascontiguousarray(dt, dtype=np.float32),
        np.ascontiguousarray(gt, dtype=np.float32)
    )
```
