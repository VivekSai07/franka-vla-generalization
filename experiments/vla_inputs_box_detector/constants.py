"""Everything the box detector fixes before it is trained (M3b spec D4, D5): the frame stride, the geometry, the augmentation, the
recipe, the gate thresholds and the presence threshold. Pure Python, no imports: the training package, the gate and the rollout
adapter (arm A5) all read these and nothing else decides them.

The detector sees each camera as follows (`inputs.preprocess`): the **front** camera as a fixed native-resolution crop of the
workspace, rows 128..415 and columns 16..495 of the 512 x 512 render (288 x 480 pixels; S0's target boxes of all 540 sealed episodes
lie within rows 175..365 and columns 51..476), the **gripper** camera area-averaged 2 x 2 to 256 x 256 (its boxes are large). A
box is S0's `(u_min, v_min, u_max, v_max, in_frame)` in [0, 1] of the full 512 x 512 frame.
"""
IMAGE = 512
OBJECTS = ("can", "bread", "lemon", "milk", "bottle", "cereal")  # props_scene.constants.OBJECTS, pinned by a test
BINS = ("blue", "orange")

# --- the cache: every STRIDE_K-th frame of every sealed episode ---------------------------------------------------------------
STRIDE_K = 8
FRAMES_PER_EPISODE = 452
CACHE_FRAMES = tuple(range(0, FRAMES_PER_EPISODE, STRIDE_K))  # 57 frames per episode
TRAIN_EPISODES, VALIDATION_EPISODES = tuple(range(480)), tuple(range(480, 540))
CACHE_DISK_GUARD_GIB = 65

# --- geometry -------------------------------------------------------------------------------------------------------------------
FRONT_CROP = (128, 416, 16, 496)  # y0, y1, x0, x1 in the 512 x 512 render
FRONT_SHAPE = (288, 480)
GRIPPER_FACTOR = 2
GRIPPER_SHAPE = (256, 256)
HEAD_STRIDE = 4
MIN_VISIBLE_PX = 2.0  # an augmented box thinner than this (input pixels) in either dimension after clipping is absent
PRESENCE_THRESHOLD = 0.5  # sigmoid of the peak heat-map logit

# --- the recipe ------------------------------------------------------------------------------------------------------------------
SEED = 20261105
EPOCHS = 60
BATCH = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
SHIFT_PROBABILITY = 0.5  # per sample and camera: translate the frame by a uniform integer shift, transform the label exactly
SHIFT_MAX_FRACTION = 0.6  # of the input's width and height
BRIGHTNESS, CONTRAST, NOISE = 0.05, 0.05, 0.01
SIZE_UNIT_PX = 16.0  # the size regression target is in units of this many input pixels
OFFSET_WEIGHT, SIZE_WEIGHT = 1.0, 1.0
FOCAL_ALPHA, FOCAL_BETA = 2.0, 4.0
MAX_PARAMETERS = 50_000_000
WIDTHS = (64, 128, 256, 512)
HEAD_CHANNELS = 128

# --- the gate (spec D5), per camera; applied before any A5 trial -----------------------------------------------------------------
GATE_MEAN_IOU = 0.85
GATE_SHARE_BELOW_HALF = 0.03  # at most this share of frames below IoU 0.5
GATE_IOU_FLOOR = 0.5
GATE_PRESENCE_F1 = 0.90  # on synthetic out-of-view frames, the smaller of the F1 of "present" and of "absent"
SYNTHETIC_SEED = 20261106
SYNTHETIC_ABSENT_FRACTION = 0.5
