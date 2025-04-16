import logging
import os

import numpy as np
import torchvision.transforms as trans
from PIL import Image

from models.PCDreamer import PCDreamer_PCN_Svd
from models.model_utils import fps_subsample
from utils.helpers import upsample_points, read_pfm
from utils.loss_utils import *


def infer_net(cfg, model=None, root='./demo_pcn/pts'):
    torch.backends.cudnn.benchmark = True

    # Setup networks and initialize networks
    if model is None:
        model = PCDreamer_PCN_Svd(cfg)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    # Inference loop
    for file_name in os.listdir(root):
        print(file_name)
        with torch.no_grad():
            partial = np.loadtxt(os.path.join(root, file_name)).astype(np.float32)
            if partial.shape[0] < cfg.CONST.N_INPUT_POINTS:
                partial = upsample_points(partial, n_points=cfg.CONST.N_INPUT_POINTS)

            partial = torch.from_numpy(partial).float().cuda()
            partial = fps_subsample(partial.unsqueeze(0), n_points=2048)

            b, n, _ = partial.shape

            view_lst = []
            img_root = os.path.join(root.replace('pts', 'imgs'), 'svd', file_name.split('.')[0])
            img_transforms = trans.Compose([
                # transforms.Scale(224),
                trans.Resize(224),
                trans.ToTensor()
            ])
            for i in range(6):
                view_path = os.path.join(img_root, 'pfm', str(i).rjust(2, '0') + '.pfm')
                image_data, _ = read_pfm(view_path)
                views = img_transforms(Image.fromarray(image_data))
                view_lst.append(views)

            mv_depth = torch.stack(view_lst).float().reshape(1, -1, 224, 224)

            pcds_pred = model(partial.contiguous(), mv_depth)

            file_str = file_name.split('.')[0]
            os.makedirs(f'./infer_res/{file_str}', exist_ok=True)
            file_name = f'./infer_res/{file_str}/' + 'pred_pcn_svd.xyz'
            # file_name1 = f'./infer_res/{file_str}/' + 'part.xyz'
            np.savetxt(file_name, pcds_pred[-1][0].cpu().numpy().squeeze())
            # np.savetxt(file_name1, partial[0].cpu().numpy().squeeze())

    return 0
