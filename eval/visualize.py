import cv2
import mmcv
import numpy as np
import pycocotools.mask as mask_util

from scipy import ndimage
# from mmdet.evaluation import get_classes
from eval import denormalize_image



def get_masks(result, num_classes=80):
    for cur_result in result:
        masks = [[] for _ in range(num_classes)]
        if cur_result is None:
            return masks

        seg_pred = cur_result[0].cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1].cpu().numpy().astype(np.int)
        cate_score = cur_result[2].cpu().numpy().astype(np.float)
        num_masks = seg_pred.shape[0]

        for idx in range(num_masks):
            cur_label = cate_label[idx]
            if cur_label == 0: continue
            else: cur_label -= 1

            cur_mask = seg_pred[idx, ...]
            rle = mask_util.encode(
                np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            rst = (rle, cate_score[idx])
            masks[cur_label].append(rst)
        return masks


def vis_seg(img_list, img_metas, result, score_thr, mask_thr, save_dir):
    assert len(img_list) == len(img_metas)
    class_names =  ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    vis_results = []
    for img, img_meta, cur_result in zip(img_list, img_metas, result):
        img = denormalize_image(img.permute(1, 2, 0).detach().cpu().numpy())
        if cur_result is None:
            print("None")
            continue

        h, w = img_meta['img_shape']
        img_show = img[:h, :w, :]

        seg_label = cur_result[0]
        seg_label = seg_label.cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1]
        cate_label = cate_label.cpu().numpy()
        score = cur_result[2].cpu().numpy()

        vis_inds = score > score_thr
        seg_label = seg_label[vis_inds]
        num_mask = seg_label.shape[0]
        cate_label = cate_label[vis_inds]
        cate_score = score[vis_inds]

        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > mask_thr).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

        seg_show = img_show.copy()
        for idx in range(num_mask):
            idx = -(idx+1)
            cur_cate = cate_label[idx]
            if cur_cate == 0: continue
            else: cur_cate -= 1

            cur_score = cate_score[idx]

            cur_mask = seg_label[idx, :,:]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
               continue
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            cur_mask_bool = cur_mask.astype(np.bool)
            seg_show[cur_mask_bool] = img_show[cur_mask_bool] * 0.5 + color_mask * 0.5

            label_text = class_names[cur_cate]
            label_text += '|{:.02f}'.format(cur_score)

            # center
            center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
            vis_pos = (max(int(center_x) - 10, 0), int(center_y))
            seg_show = cv2.putText(seg_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
        mmcv.imwrite(cv2.cvtColor(seg_show, cv2.COLOR_BGR2RGB), '/content/drive/MyDrive/SOLO-implementation/debug/{}.jpg'.format( img_meta['image_id']))

        # mmcv.imwrite(cv2.cvtColor(seg_show, cv2.COLOR_BGR2RGB), '/content/drive/MyDrive/SOLO-implementation/debug/{}.jpg'.format( img_meta['image_id']))
     
        cv2.waitKey(0)
        vis_results.append(seg_show)
        
    return vis_results