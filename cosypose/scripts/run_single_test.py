
from PIL import Image
import numpy as np
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse
import cv2
import glob
import os
import time

from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset

# Pose estimator
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse
from cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper

# Rendering
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.visualization.singleview import render_prediction_wrt_camera
from cosypose.visualization.plotter import Plotter
from bokeh.io import show, output_notebook; output_notebook()
from bokeh.io import export_png

# Detection
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from cosypose.integrated.detector import Detector

from cosypose.evaluation.pred_runner.bop_predictions import BopPredictionRunner

from cosypose.utils.distributed import get_tmp_dir, get_rank
from cosypose.utils.distributed import init_distributed_mode

from cosypose.config import EXP_DIR, RESULTS_DIR

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_detector(run_id):
    run_dir = EXP_DIR / run_id
    # print(run_dir)
    # import pdb
    # pdb.set_trace()
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model

def load_pose_models(coarse_run_id, refiner_run_id=None, n_workers=8):
    run_dir = EXP_DIR / coarse_run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_pose(cfg)
    #object_ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_cad')
    object_ds = make_object_dataset(cfg.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_model = load_model(coarse_run_id)
    refiner_model = load_model(refiner_run_id)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
                                      refiner_model=refiner_model)
    return model, mesh_db

def getModel(): 
    # load models (synt + real) jaehyung's default
    detector_run_id='detector-bop-ycbv-synt+real--292971'
    coarse_run_id='coarse-bop-ycbv-synt+real--822463'
    refiner_run_id='refiner-bop-ycbv-synt+real--631598'

    # load models (PBR) 
    # detector_run_id='detector-bop-ycbv-pbr--970850'
    # coarse_run_id='coarse-bop-ycbv-pbr--724183'
    # refiner_run_id='refiner-bop-ycbv-pbr--604090'

    detector = load_detector(detector_run_id)
    pose_predictor, mesh_db = load_pose_models(coarse_run_id=coarse_run_id,refiner_run_id=refiner_run_id,n_workers=4)
    return detector,pose_predictor


def inference(detector,pose_predictor,image,camera_k):

    print("enter here--------------------1")
    # import pdb
    #[1,540,720,3]->[1,3,540,720]
    images = torch.from_numpy(image).cuda().float().unsqueeze_(0)
    images = images.permute(0, 3, 1, 2) / 255
    #[1,3,3]
    cameras_k = torch.from_numpy(camera_k).cuda().float().unsqueeze_(0)
    #2D detector 
    #print("start detect object.")
    box_detections = detector.get_detections(images=images, one_instance_per_class=True, 
                    detection_th=0.9,output_masks=True, mask_th=0.9)
    # pdb.set_trace()
    # box_detections = detector.get_detections(images=images, one_instance_per_class=True, 
    #                 detection_th=0.8,output_masks=True, mask_th=0.7)
    
    # box_detections = detector.get_detections(images=images, one_instance_per_class=True, 
    #                 detection_th=0.7,output_masks=True, mask_th=0.7)
    
    # print('box_detections')
    # print(box_detections)


    #pose esitimition
    if len(box_detections) == 0:
        return None, None
    #print("start estimate pose.")
    # final_preds, all_preds=pose_predictor.get_predictions(images, cameras_k, detections=box_detections,
    #                     n_coarse_iterations=1,n_refiner_iterations=1)
    final_preds, all_preds=pose_predictor.get_predictions(images, cameras_k, detections=box_detections,
                        n_coarse_iterations=1,n_refiner_iterations=5)
    #print("inference successfully.")
    #result: this_batch_detections, final_preds
    # print("final preds & box detection in inference")
    # print(final_preds.cpu())
    # print(box_detections.cpu())

    return final_preds.cpu(), box_detections.cpu()


def main():
    print("start...........................................")
    figures = dict()
    plotter = Plotter()
    detector,pose_predictor=getModel()
    # camera_k = np.array([[1066.778,    0,      312.9869 ],\
    #                     [  0,      1067.487, 241.3109],\
                        # [  0,        0,        1,     ]])

    # camera_k = np.array([[519.70,    0,      317.51 ],\
    #                     [  0,      520.03, 235.35],\
    #                     [  0,        0,        1,     ]])

    # camera_k = np.array([[577.30,    0,      317.51 ],\
    #                     [  0,      577.30, 235.35],\
    #                     [  0,        0,        1,     ]])
    

    # camera_k = np.array([[590.00,    0,      317.51 ],\
    #                     [  0,      590.00, 235.35],\
    #                     [  0,        0,        1,     ]])0
    
    # 240814 exp    
    # 527.0865605963895, 525.9207121379819, 313.76590897104717, 237.89436263132148
    # camera_k = np.array([[527.1667,    0,      313.2282 ],\
    #                     [  0,      526.4720,   236.1043],\
    #                     [  0,        0,        1,     ]])


    # zed camera intrinsic
    # camera_k = np.array([[352.0156,    0,      318.1740 ],\
    #                     [  0,      351.6434,   170.5769],\
    #                     [  0,        0,        1,     ]])
    
    
    # zed camera intrinsic
    camera_k = np.array([[347.9726,    0,      318.7975 ],\
                        [  0,      346.9995,   170.9622],\
                        [  0,        0,        1,     ]])
    
    



    # 250120 exp
    # base_path = "/media/nesl/T9/NRF_exp/250120/raw_images/exp_010/"
    # save_path = '/media/nesl/T9/NRF_exp/250120/outputs/exp_010/singleview_refine/'
    # save_path_2d = '/media/nesl/T9/NRF_exp/250120/outputs/exp_010/2d_th07_refine/'

    # 250123 exp
    # base_path = "/media/nesl/T9/0120_NRF/250123/exp_001/raw_images/"
    # save_path = '/media/nesl/T9/0120_NRF/250123/exp_001/cosypose/singleview_refine/'
    # save_path_2d = '/media/nesl/T9/0120_NRF/250123/exp_001/cosypose/2d_th07_refine/'

    # 250908_607
    # base_path = "/media/nesl/3C3CCD603CCD15B4/251127/raw_images/exp_006/color/"
    # save_path = '/media/nesl/3C3CCD603CCD15B4/251127/obj_output/exp_006/singleview_refine/'
    # save_path_2d = '/media/nesl/3C3CCD603CCD15B4/251127/obj_output/exp_006/2d_th09_refine/'

    # 251127_301
    # base_path = "/media/nesl/3C3CCD603CCD15B4/251127_v1/raw_images/exp_003/color/"
    # save_path = '/media/nesl/3C3CCD603CCD15B4/251127_v1/obj_output/exp_003/singleview_refine/'
    # save_path_2d = '/media/nesl/3C3CCD603CCD15B4/251127_v1/obj_output/exp_003/2d_th09_refine/'
    
    # the target image
    # base_path = "/media/nesl/T9/NRF_exp/240116/test_30000_original/test_30000_img"
    # base_path = "/media/nesl/T9/NRF_exp/240814/raw_images/exp_004/"
    # base_path = "/home/nesl/sohee/1031_raw_images/L_down_001/"
    # base_path = "/media/nesl/hdd0/YCB-Video/YCB_Video_Dataset/data/0088"
    
    # save_path = '/media/nesl/hdd0/YCB-Video_cosypose/results_0017/singleview_norefine/'
    # save_path_2d = '/media/nesl/hdd0/YCB-Video_cosypose/results_0017/2d_th07_norefine/'

    # save_path = '/home/nesl/sohee/1031_outputs/L_down_001/singleview_refine/'
    # save_path_2d = '/home/nesl/sohee/1031_outputs/L_down_001/2d_th07_refine/'
    
    # 20250112 tykim
    base_path    = "/home/nesl/Documents/cosypose/data/raw_images/exp_003/color/"
    save_path    = "/home/nesl/Documents/cosypose/data/outputs/exp_003/singleview_refine/"
    save_path_2d = "/home/nesl/Documents/cosypose/data/outputs/exp_003/2d_th09_refine/"
    
    files = sorted( glob.glob(os.path.join(str(base_path),'*-color.png')) )
    


    # Rendering of the predicted object
    urdf_ds_name = 'ycbv'
    renderer = BulletSceneRenderer(urdf_ds_name)
    input_dim = (640, 480)
    # input_dim = (640, 360)
    cam = dict(
        resolution=input_dim,
        K=camera_k,
        TWC=np.eye(4)
    )

    print('Enter here-----------------------------')


    for file in files:
        print('image = ',file)
        img = Image.open(file)
        # img_cropped = img.crop((80,0, 560, 360))
        # img_resized = img_cropped.resize((640, 480))
        img = np.array(img)

        img = cv2.copyMakeBorder(img, 60, 60, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # resize zed image [640, 360] -> [640, 480]
        # img = cv2.resize(img, (640, 480))

        # Start time measurement
        start_time = time.time()

        # predict
        pred, pred_2d = inference(detector, pose_predictor, img, camera_k)

        # End time measurement
        end_time = time.time()

        # Calculate inference time
        inference_time = end_time - start_time
        print(f"Infer time: {inference_time:.4f} s")
        print("----------------------------------------------------")

        # print('pred is ', '\n')
        # print(pred, '\n')


        # print('2d pred is ', '\n')
        # print(pred_2d, '\n')

        # print('pred poses')
        # print(pred.poses)

        # print('pred K_crop')
        # print(pred.K_crop)

        # for rendering
        if pred != None:
            # pred_rendered = render_prediction_wrt_camera(renderer, pred, cam)
            # overlay = plotter.plot_overlay_array(img, pred_rendered)

            # img_draw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # overlay_draw = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            # cv2.imshow('input image', img_draw)
            # cv2.imshow('input + pred image', overlay_draw)
            # cv2.waitKey(10)

            # # Save results
            # pose_path = save_path + 'poses/' + file[-16:-10] + '.npy'
            # img_path = save_path + 'images/' + file[-16:-10] + '.png'
            # cv2.imwrite(img_path, overlay_draw)

            # Report results: poses,poses_input,K_crop,boxes_rend,boxes_crop
            print("num of pred:",len(pred))
            for i in range(len(pred)):
                print("object ",i,":",pred.infos.iloc[i].label,"------\n  pose:",pred.poses[i].numpy(),"\n  detection score:",pred.infos.iloc[i].score)

                # # if pred.infos.iloc[i].label == 'obj_000014':
                # save result
                pose_path = save_path + 'poses/' + pred.infos.iloc[i].label + '_' + file[-16:-10] + '.npy'
                np.save(pose_path, pred.poses[i].numpy())

                pose_path = save_path_2d + 'masks/' + pred.infos.iloc[i].label + '_' + file[-16:-10] + '.npy'
                np.save(pose_path, pred_2d.masks[i].numpy())
        # 2d data extraction







    renderer.disconnect()
    print("end")

if __name__ == '__main__':
    main()

