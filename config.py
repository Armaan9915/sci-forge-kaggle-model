import torch

class Config:
    # ==================== PATHS ====================
    TRAIN_AUTHENTIC_DIR = "AUTHENTIC_TRAINING_IMAGES_PATH"
    TRAIN_FORGED_DIR = "FORGED_TRAINING_IMAGES_PATH"
    TRAIN_MASK_DIR = "TRAIN_MASKS_PATH"
    
    TEST_AUTHENTIC_DIR = "AUTHENTIC_TEST_IMAGES_PATH"
    TEST_FORGED_DIR = "FORGED_TEST_IMAGES_PATH"
    TEST_MASK_DIR = "TEST_MASKS_PATH"

    HQ_AUTHENTIC_DIR = "HQ_AUTHENTIC_IMAGES_PATH"
    HQ_FORGED_DIR = "HQ_FORGED_IMAGES_PATH"
    HQ_MASK_DIR = "HQ_MASKS_PATH"
    
    OUTPUT_DIR = "OUTPUTS_PATH"
    CHECKPOINT_DIR = "CHECKPOINTS_PATH"
    
    # DINOv2 Model Path
    DINOV2_PATH = "DINOV2_PATH"

    # ==================== MODEL ====================
    INPUT_SIZE = 462
    PATCH_SIZE = 14
    FEATURE_MAP_SIZE = 33
    EVAL_MASK_SIZE = 224
    
    DINO_EMBEDDING_DIM = 768
    MLP_HIDDEN_DIM = 1024
    FINAL_EMBEDDING_DIM = 512
    
    MAX_INSTANCES = 5
    
    # ==================== TRAINING ====================
    NUM_EPOCHS = 40
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 2
    
    ENCODER_FREEZE_EPOCHS = 10
    
    # Learning Rates
    MLP_LEARNING_RATE = 1e-3
    DINO_LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    
    T_MAX = NUM_EPOCHS
    ETA_MIN = 1e-7
    USE_AMP = True
    
    # ==================== LOSS WEIGHTS (UPDATED) ====================
    # SupCon is the primary driver now. 
    # Separation helps keep centroids apart. Uniformity helps within-cluster density.
    LOSS_WEIGHT_SUPCON = 1.2        # Main clustering force
    LOSS_WEIGHT_SEPARATION = 1.0    # Helper: forces inter-cluster distance
    LOSS_WEIGHT_UNIFORMITY = 0.5    # Helper: penalizes loose clusters
    
    # SupCon Hyperparameters
    SUPCON_TEMPERATURE = 0.07       # Crucial: Controls cluster tightness. 0.07 is standard.
    
    # Other Loss Hyperparameters
    UNIFORMITY_THRESHOLD = 0.1
    SEPARATION_MARGIN = 2.0
    
    # ==================== AUGMENTATION ====================
    AUG_ROTATION_LIMIT = 30
    AUG_PROB_FLIP = 0.5
    AUG_PROB_ROTATE = 0.5
    
    AUG_BRIGHTNESS = 0.25
    AUG_CONTRAST = 0.15
    AUG_SATURATION = 0.15
    AUG_HUE = 0.05
    AUG_PROB_COLOR = 0.5
    
    AUG_JPEG_QUALITY_RANGE = (80, 100)
    AUG_PROB_JPEG = 0.4
    
    AUG_BLUR_LIMIT = (3, 5)
    AUG_PROB_BLUR = 0.3
    
    AUG_NOISE_VAR_LIMIT = (5.0, 15.0)
    AUG_PROB_NOISE = 0.3
    
    # ==================== INFERENCE ====================
    HDBSCAN_MIN_CLUSTER_SIZE = 15
    HDBSCAN_MIN_SAMPLES = 5
    HDBSCAN_METRIC = 'euclidean'
    HDBSCAN_CLUSTER_SELECTION_METHOD = 'eom'
    
    # Geometry validation (Cosine Distance > Threshold to be forgery)
    GEOMETRY_THRESHOLD = 0.5
    
    # ==================== VALIDATION ====================
    VAL_SAMPLE_RATIO = 0.3
    VAL_AUTHENTIC_RATIO = 0.3
    VAL_FORGED_RATIO = 0.7
    VAL_FREQUENCY = 3
    
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # ==================== TUNING ====================
    TUNING_SAMPLE_RATIO = 0.20
    TUNING_AUTHENTIC_RATIO = 0.30
    TUNING_FORGED_RATIO = 0.70
    
    TEST_MIN_CLUSTER_SIZES = [10, 15, 20, 25]
    TEST_MIN_SAMPLES = [3, 5, 7, 10]
    TEST_GEOMETRY_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
    USE_HIERARCHICAL_SEARCH = True
    
    # ==================== SYSTEM ====================
    SAVE_TOP_PREDICTIONS = 25
    WORLD_SIZE = 2
    BACKEND = "nccl"
    
    # ==================== SMOKE TEST ====================
    SMOKE_TEST = False
    SMOKE_TEST_SAMPLES = 30
    SMOKE_TEST_EPOCHS = 5
    SMOKE_TEST_BATCH_SIZE = 4
    SMOKE_TEST_FREEZE_EPOCHS = 1
    SMOKE_TEST_GRAD_ACCUM = 1
    
    @classmethod
    def set_smoke_test(cls, enable=True):
        cls.SMOKE_TEST = enable
        if enable:
            cls.NUM_EPOCHS = cls.SMOKE_TEST_EPOCHS
            cls.ENCODER_FREEZE_EPOCHS = cls.SMOKE_TEST_FREEZE_EPOCHS
            cls.T_MAX = cls.SMOKE_TEST_EPOCHS
            cls.BATCH_SIZE = cls.SMOKE_TEST_BATCH_SIZE
            cls.GRADIENT_ACCUMULATION_STEPS = cls.SMOKE_TEST_GRAD_ACCUM
            print(f"🔥 SMOKE TEST MODE ENABLED")
    
    @classmethod
    def print_config(cls):
        print("=" * 60)
        print("DINOv2 + MLP + SUPCON COPY-MOVE DETECTION")
        print("=" * 60)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key}: {value}")
        print("=" * 60)