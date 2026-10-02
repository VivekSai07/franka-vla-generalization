FROM sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
# robosuite 1.5.2 requires mujoco>=3.3. Re-pin 3.13.0 so this layer cannot upgrade it.
RUN pip install --no-cache-dir "robosuite==1.5.2" "mujoco==3.13.0"
