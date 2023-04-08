from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import skimage.io as io
import numpy as np
import torch
import matplotlib.pyplot as plt

COCO_CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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


class CocoV2(Dataset):
  CLASSES = COCO_CLASSES
  def __init__(self,annotations_file, transforms=None, target_transform=None):
    self.annotations_file = annotations_file
    self.coco = COCO(self.annotations_file)

    whole_image_ids = list(sorted(self.coco.imgs.keys()))
    self.ids = []
    self.no_anno_list = []

    # remove img without bbox or anno
    for idx in whole_image_ids:
      annotations_ids = self.coco.getAnnIds(imgIds=idx, iscrowd=False)
      # return annotations , can have many
      annotations = self.coco.loadAnns(annotations_ids)
      if len(annotations_ids) == 0:
        self.no_anno_list.append(idx)
      if self._has_only_empty_bbox(annotations):
        self.no_anno_list.append(idx)
      else:
        self.ids.append(idx)

    # since coco catId is not continuous
    self.coco_cat_ids = sorted(self.coco.getCatIds())
    self.coco_cat_ids_to_continuous_ids = {coco_id: i+1 for i, coco_id in enumerate(self.coco_cat_ids)}

    # map cat id to label
    self.coco_ids_to_class_names  = {category['id']: category['name'] for category in self.coco.loadCats(self.coco_cat_ids)}

    self.transforms = transforms

  def __len__(self):
      return len(self.ids)
  
  def showImg(self,image_id):
    img = self.coco.loadImgs(image_id)[0]
    I = io.imread(img['coco_url'])
    plt.axis('off')
    plt.imshow(I)
    plt.show()

  # check if the bbox is empty
  def _has_only_empty_bbox(self, annotations):
      for annot in annotations:
        if annot["bbox"] == []:
          return True
        for o in annot['bbox'][2:]:
          if o <= 1:
            return True
  def _load_image(self, image_id):
    img = self.coco.loadImgs(image_id)[0]
    img = io.imread(img['coco_url'])
    return img 

  def __getitem__(self, index):
    image_id = self.ids[index]
    image = self._load_image(image_id)
    target = self._load_target(image_id)
    # image.shape = (230, 352, 3) , ori_shape = (230, 352)
    img_meta = dict(ori_shape=image.shape[:2], image_id=image_id)


    if self.transforms is not None:
      transformed = self.transforms(image=image,masks=target['masks'], bboxes=target['boxes'],category_ids=target['labels'])
      image = transformed['image']
      # update img after transform
      img_meta['img_shape'] = image.shape[-2:]
      target['masks'] = transformed['masks']
      target['boxes'] = transformed['bboxes']

    # print(target['masks'])
    # print(target['boxes'])
    target['image_id'] = torch.as_tensor(target['image_id'], dtype=torch.int32)
    target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
    # print(target['masks'])
    # target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8)
    # print(np.array(target['masks']))
    # target['masks'] = torch.from_numpy(np.array(target['masks']))
    print("stack mask")
    target['masks'] = torch.stack(target['masks'])
    # print(target['masks'].shape)
    target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
    target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
    target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.uint8)
    # print(target['boxes'].shape)
    return image,target ,img_meta



  def _load_target(self, image_id):
        annot_ids = self.coco.getAnnIds(image_id)

        # frame to store boxes, many boxes per img
        boxes = np.zeros((0, 4))
        # frame to store label, many label per img
        labels = np.zeros((0, 1))

        annots = self.coco.loadAnns(annot_ids)

        for annot in annots:
            box = np.zeros((1, 4))
            label = np.zeros((1, 1))

            box[0, :4] = annot['bbox']
            label[0, :1] = self.coco_cat_ids_to_continuous_ids[annot['category_id']]

            boxes = np.append(boxes, box, axis=0)
            labels = np.append(labels, label, axis=0)
        labels = labels.ravel()
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = np.array([annot['area'] for annot in annots], dtype=np.float32)
        iscrowd = np.array([annot['iscrowd'] for annot in annots], dtype=np.uint8)

        # Strangely, when only the mask is converted to NumPy, an error occurs.
        # It seems that albumentations does not support it.
        masks = [self.coco.annToMask(annot) for annot in annots]
        # print([x.shape for x in masks])
        # masks = torch.LongTensor(np.max(np.stack([self.coco.annToMask(ann) * ann["category_id"] 
        #                                          for ann in annots]), axis=0)).unsqueeze(0)
        # print(masks[0,...].shape)
        return {'image_id': torch.tensor([image_id]), 'boxes': boxes, 'masks': masks, 'labels': labels, 'area': area, 'iscrowd': iscrowd}