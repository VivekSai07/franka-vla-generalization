import numpy as np
rng = np.random.default_rng(7)
def wilson_lo(k, n, z=1.96):
    n = np.maximum(n, 1); ph = k/n
    return (ph + z*z/(2*n) - z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n))) / (1 + z*z/n)
def power_one(L, s, p, reps=200000, floor=30):
    n = rng.binomial(L, s, reps); k = rng.binomial(n, p)
    ok = (n >= floor) & (wilson_lo(k, n) > 0.5)
    return ok.mean(), (n >= floor).mean()
print('single held-out object: P(lower bound of bin-choice accuracy > 0.5 and >=30 stable placements)')
print('stable-placement prob s =', 0.85, '(bottle .93, cereal .86 in P2; lower values below)')
print(' L/object |  p=0.55  0.60  0.65  0.70  0.75  0.80  0.90 | P(floor met)')
for L in (20, 40, 60, 80, 120):
    row = [power_one(L, 0.85, p)[0] for p in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90)]
    print(f'{L:8d} | ' + ' '.join(f'{x:5.2f}' for x in row) + f' | {power_one(L,0.85,0.7)[1]:.3f}')
print('\nBOTH objects (independent) at the same true p:')
print(' L/object |  p=0.65  0.70  0.75  0.80  0.90')
for L in (20, 40, 60, 80, 120):
    row = [power_one(L, 0.85, p)[0]**2 for p in (0.65, 0.70, 0.75, 0.80, 0.90)]
    print(f'{L:8d} | ' + ' '.join(f'{x:5.2f}' for x in row))
print('\nfalse positive when the policy is at chance (p=0.5), both objects:', [round(power_one(L, 0.85, 0.5)[0]**2, 5) for L in (20, 60, 120)])
print('\nfloor sensitivity: P(>=30 stable placements) if stable prob is lower (s=0.6, 0.5):', {L: (round(power_one(L,0.6,0.7)[1],3), round(power_one(L,0.5,0.7)[1],3)) for L in (60,80,120)})
# trained-object gate: wrong-bin point estimate <= 5% among stable placements, 80 trained layouts, stable prob 0.85
def gate(Lt, s, w, reps=200000):
    n = rng.binomial(Lt, s, reps); k = rng.binomial(n, w); return ((k/np.maximum(n,1)) <= 0.05).mean()
print('\ntrained gate pass probability (80 layouts, s=.85) for true wrong-bin rate w:', {w: round(gate(80,.85,w),3) for w in (0.0,0.01,0.02,0.04,0.05,0.08,0.10)})
