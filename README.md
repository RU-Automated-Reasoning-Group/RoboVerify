# RoboVerify

best kl = -0.83 [ 0.00287604  0.05722181  0.13267801  0.01365238 -0.12319135  0.11372262  -0.04297739  0.01262963  0.07352702]
average: -1.8290530378267873, [ 0.04924645  0.04736446  0.09046513 -0.03631592 -0.05455689  0.08911783, -0.02385015  0.02479539  0.12097145]


best kl -0.3074323277065894 [ 0.0279608   0.04278932  0.13399421  0.02289095 -0.04434892  0.13288633 -0.01862492  0.00964001  0.05832452]
average:  -1.012017127608849 [ 0.022194    0.05027753  0.10043376 -0.01963008 -0.04812001  0.09452236 -0.03093935  0.01319206  0.09157777]

scores [np.float64(0.8815281749273863), np.float64(0.889955072936398), np.float64(0.9146410798856522), np.float64(0.9153232484390901), np.float64(0.9166270316713886), np.float64(0.9181182841606098), np.float64(0.9185416337450246), np.float64(0.9303997201800137), np.float64(0.93880064346262), np.float64(0.9515211107895102), np.float64(0.9698067773191009), np.float64(0.9718758469370651), np.float64(0.9832462354761783), np.float64(0.9840943306836354), np.float64(0.9858267160398193), np.float64(1.0189825567741226)]
average of best k 0.9930374597434389
best mu [ 0.02423896  0.05111898  0.12557382 -0.00520431  0.00344668  0.15214003
 -0.00435389  0.00562681  0.14773177]
next mu: 

Reverse program
```python
def reverse(n0):
    t = tbl
    while exists(x.):
        put(x, t) # put x on t
        t = x
    put(n0, t)
```

For this program, we need to systematically analyze if `t` is null for each of its appearence.

## Up-to-date files
reverse: test_reverse_include_tbl.py
unstack: test_unstack_single_tower_b0_bottom_exists_top.py
stack: 

## define a scattered relation that are true when two blocks are both on_table

## in the low level verification, we will translate this to boxes are far apart from each other.