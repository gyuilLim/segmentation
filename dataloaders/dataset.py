import os
import numpy as np
from PIL import Image
from torch.utils import data
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'
        elif dataset == 'cityscapes':
            return '/home/vision/gyuil/lab/Segmentation/cityscapes'
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

class CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 19

    def __init__(self, base_size, crop_size, root=Path.db_root_dir('cityscapes'), split="train"):
        self.root = root
        self.split = split
        self.base_size = base_size 
        self.crop_size = crop_size
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = np.array(Image.open(img_path).convert('RGB'))
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)

        if self.split == 'train':
            sample = self.transform_tr(_img, _tmp)
        elif self.split == 'val':
            sample = self.transform_val(_img, _tmp)
        elif self.split == 'test':
            sample = self.transform_ts(_img, _tmp)

        return sample['image'], sample['mask']

    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, image, mask):
        transform = A.Compose([
            A.RandomResizedCrop(height=self.crop_size[0], width=self.crop_size[1], scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        return transform(image=image, mask=mask)

    def transform_val(self, image, mask):
        transform = A.Compose([
            A.Resize(height=self.crop_size[0], width=self.crop_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        return transform(image=image, mask=mask)
    
    def transform_ts(self, image, mask):
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        return transform(image=image, mask=mask)
