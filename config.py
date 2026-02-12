DATA_TRAIN_DIR = "data/train"
DATA_VAL_DIR = "data/val"
DATA_TEST_DIR = "data/test"
DATA_TEST_OTHER_DIR = "data/test_other"
DESTRUCTION = "noise"
IMG_SIZE = 128
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.01
MODEL_PATH = "model.pth"
RESULTS_DIR = "results"

# Face alignment (detection + 5 landmarks)
USE_ALIGNMENT = False
# Bonus: identity loss (keep face identity)
USE_IDENTITY_LOSS = True
IDENTITY_LOSS_WEIGHT = 0.1