import logging

from tqdm import tqdm

import utils.data_loaders
import utils.helpers
from models.PCDreamer import PCDreamer_PCN_Svd
from utils.average_meter import AverageMeter
from utils.loss_utils import *


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    torch.backends.cudnn.benchmark = True
    logging.getLogger('PIL').setLevel(logging.WARNING)

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.TEST),
            batch_size=1,
            num_workers=cfg.CONST.NUM_WORKERS // 2,
            collate_fn=utils.data_loaders.collate_fn_rgb,
            pin_memory=True,
            shuffle=False)

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

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['CD', 'DCD', 'F1'])
    test_metrics = AverageMeter(['CD', 'DCD', 'F1'])
    category_metrics = dict()
    print("test_datasets: ", n_samples)

    # Testing loop
    with tqdm(test_data_loader) as t:
        for model_idx, (taxonomy_id, model_id, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                gt = data['gtcloud']
                multirgb = data['views'].reshape(gt.shape[0], -1, 224, 224)

                # partial = partial.repeat(8, 1, 1)
                # gt = gt.repeat(8, 1, 1)
                # multirgb = multirgb.repeat(8, 1, 1, 1)

                b, n, _ = partial.shape

                pcds_pred = model(partial, multirgb)
                # print("sdffffffffffffffffffffffffffffffff",pcds_pred[-1].shape)
                cdl1, cdl2, f1 = calc_cd(pcds_pred[-1], gt, calc_f1=True)
                dcd, _, _ = calc_dcd(pcds_pred[-1], gt)
                # cdl1, cdl2, f1 = calc_cd(pcds_pred[0], gt, calc_f1=True)
                # dcd, _, _ = calc_dcd(pcds_pred[0], gt)

                cd = cdl1.mean().item() * 1e3
                dcd = dcd.mean().item()
                f1 = f1.mean().item()

                _metrics = [cd, dcd, f1]
                test_losses.update([cd, dcd, f1])

                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(['CD', 'DCD', 'F1'])
                category_metrics[taxonomy_id].update(_metrics)

                t.set_description('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                                  (model_idx + 1, n_samples, taxonomy_id, model_id,
                                   ['%.4f' % l for l in test_losses.val()
                                    ], ['%.4f' % m for m in _metrics]))

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    print('Epoch ', epoch_idx, end='\t')
    for value in test_losses.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/cd', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/dcd', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/f1', test_losses.avg(2), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(0)
