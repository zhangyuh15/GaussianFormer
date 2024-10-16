# try:
#     from vis import save_occ
# except:
#     print('Load Occupancy Visualization Tools Failed.')
import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")

class TOSAVE(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
    
    def forward(self, img,projection_mat, image_wh, occ_xyz, occ_label, occ_cam_mask):
        meta=dict()
        meta["projection_mat"] = projection_mat
        meta["image_wh"] = image_wh
        meta["occ_xyz"] = occ_xyz
        meta["occ_label"] = occ_label
        meta["occ_cam_mask"] = occ_cam_mask

        return self.f(imgs=img, metas=meta)
    
def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP

    distributed = False
    world_size = 1
    
    writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    
    my_model = my_model.cuda()

    to_save = TOSAVE(my_model)

    image = torch.randn([1, 6, 3, 704, 256]).cuda() # 
    metas = dict()
    # metas["projection_mat"] = torch.randn([6, 1 , 24, 4])# .cuda()
    # metas["image_wh"] = None # torch.Tensor((704, 256)).cuda()
    # metas['occ_xyz'] =  torch.randn([1, 1, 1, 3])# .cuda()
    # metas['occ_label'] =  torch.randn([1, 1, 1, 1])# .cuda()
    # metas['occ_cam_mask'] =  torch.randn([1, 1, 1, 1])# .cuda()
    projection_mat= torch.randn([6, 1 , 24, 4]).cuda()
    image_wh = None # torch.Tensor((704, 256)).cuda()
    occ_xyz =  torch.randn([1, 1, 1, 3]).cuda()
    occ_label =  torch.randn([1, 1, 1, 1]).cuda()
    occ_cam_mask =  torch.randn([1, 1, 1, 1]).cuda()


    # oout = my_model(imgs=image, metas=metas)
    # print(type(my_model))
    # for i, m in enumerate(my_model.modules()):
    #     if i in [0, 1]:
    #         continue
    #     print(i, ": ", )
    #     print(type(m))
    #     kk = m.cpu()
    # for i, m in enumerate(my_model.children()):
    #     print(i, ": ", )
    #     kk = m.cpu()
    #     print(kk)
    # print(oout)
    # torch.onnx.export(my_model, (image, metas), "ddp.onnx", input_names=['input', "input1"],
    #                 output_names=['output'], opset_version=11)
    all_input = (image,projection_mat, image_wh, occ_xyz, occ_label, occ_cam_mask)
    torch.onnx.export(to_save, all_input, "ddp.onnx", input_names=['input0', "input1", "input2", "input3", "input4", "input5"],
                     output_names=['output'])
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/nuscenes_gs25600_solid.py')
    parser.add_argument('--work-dir', type=str, default='./out')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    args = parser.parse_args()
    
    ngpus = 1 # torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    # if ngpus > 1:
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    # else:
    main(0, args)
