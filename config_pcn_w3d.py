from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.N_RENDERINGS               = 6
__C.DATASETS.SHAPENET.N_POINTS                   = 2048
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = './dataset/PCN/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = './dataset/PCN/%s/complete/%s/%s.pcd'
__C.DATASETS.SHAPENET.VIEW_PATH                  = './datasets/PCN_imgs_w3d/%s/%s/'

#
# Dataset
#
__C.DATASET                                      = edict()
__C.DATASET.TRAIN_DATASET                        = 'ShapeNetW3d'
__C.DATASET.TEST_DATASET                         = 'ShapeNetW3d'
__C.DATASET.VAL_DATASET                         = 'ShapeNetW3d'

#
# Constants
#
__C.CONST                                        = edict()

__C.CONST.NUM_WORKERS                            = 4
__C.CONST.N_INPUT_POINTS                         = 2048

#
# Directories
#

__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = 'PCDreamer_PCN_W3d'
__C.CONST.DEVICE                                 = '0,1,2,3'
__C.CONST.WEIGHTS                                = './checkpoints/PCN_w3d.pth'

# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.N_SAMPLING_POINTS                    = 2048
__C.NETWORK.step1                    = 4
__C.NETWORK.step2                    = 8
__C.NETWORK.merge_points = 512
__C.NETWORK.local_points = 512
__C.NETWORK.view_distance = 0.7

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 24
__C.TRAIN.N_EPOCHS                               = 400
__C.TRAIN.SAVE_FREQ                              = 50
__C.TRAIN.LEARNING_RATE                          = 0.0001
__C.TRAIN.LR_MILESTONES                          = [50, 100, 150, 200, 250]
__C.TRAIN.LR_DECAY_STEP                          = [40,80,120,160,200,240,280,320,360]
__C.TRAIN.WARMUP_STEPS                           = 300
__C.TRAIN.GAMMA                                  = 0.7
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
