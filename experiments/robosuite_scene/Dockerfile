FROM sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
# robosuite 1.5.2 does not run on MuJoCo 3.13.0 (binding_utils enum asserts, data.qM removed).
# --no-deps keeps mink (numpy<2) and opencv-python out; only what robosuite imports is added.
RUN pip install --no-cache-dir --no-deps "robosuite==1.5.2" "mujoco==3.4.0" \
 && pip install --no-cache-dir "numba==0.68.0" "scipy==1.15.3" "termcolor==3.3.0" "numpy==2.2.6"
