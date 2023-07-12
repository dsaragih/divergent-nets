
import pandas as pd
import albumentations as albu
import cv2
from torch.utils.data import DataLoader
import os
import numpy as np
from data.dataset import Dataset

def df_from_csv_file_array(csv_file_arrya):

    df =pd.DataFrame(columns=["image_path", "mask_path"])

    for csv in csv_file_arrya:
        temp_df = pd.read_csv(csv)

        df = pd.concat([df, temp_df], ignore_index=True)

    return df

def df_from_img_dir(img_dir_path):
    # img_dir_path contains two directories: images and masks
    # each directory contains images with the same name
    # e.g. img_dir_path/images/1.png and img_dir_path/masks/1.png
    # will be paired together in the dataframe

    df = pd.DataFrame(columns=["image_path", "mask_path"])
    mask_path = os.path.join(img_dir_path, "masks")
    for img in os.listdir(os.path.join(img_dir_path, "masked-images")):
        new_row = pd.Series({"image_path": os.path.join(img_dir_path, "masked-images", img), "mask_path": os.path.join(mask_path, img)})
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    
    return df

def df_from_npz(npz_path):
    """npz_path contains a numpy array of shape (N, H, W, 4), where
    the last dimension is the mask. The first three dimensions are the image.

    First we output .jpg images to a directory, then we use df_from_img_dir
    """
    df = pd.DataFrame(columns=["image_path", "mask_path"])
    npz = np.load(npz_path)["arr_0"]

    for i in range(npz.shape[0]):
        img = npz[i, :, :, :3]
        mask = npz[i, :, :, 3]
        img_path = os.path.join(npz_path, "masked-images", str(i) + ".jpg")
        mask_path = os.path.join(npz_path, "masks", str(i) + ".jpg")
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)
        new_row = pd.Series({"image_path": img_path, "mask_path": mask_path})
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        albu.Resize(height=256, width=256, always_apply=True),

        #albu.IAAAdditiveGaussianNoise(p=0.2),
        #albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),

        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
    ]
    return albu.Compose(train_transform)



def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        albu.Resize(height=256, width=256, always_apply=True),
    ]
    return albu.Compose(test_transform)



def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')



def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = []
    if preprocessing_fn:
        _transform.append(albu.Lambda(image=preprocessing_fn))
    _transform.append(albu.Lambda(image=to_tensor, mask=to_tensor))

    
    return albu.Compose(_transform)



def prepare_data(opt, preprocessing_fn):


    train_val_df = df_from_img_dir(opt.img_dir)
    train_df = train_val_df.sample(frac=0.8, random_state=0)
    val_df = train_val_df.drop(train_df.index)
    # train_df = df_from_csv_file_array(opt.train_CSVs)
    # val_df = df_from_csv_file_array(opt.val_CSVs)


    train_dataset = Dataset(
        train_df,
        grid_sizes=opt.grid_sizes_train,
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=opt.classes,
        pyra = opt.pyra
    )

    valid_dataset = Dataset(
        val_df, 
        grid_sizes=opt.grid_sizes_val,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=opt.classes,
        pyra = opt.pyra
    )
    
    # opt.bs = 16, opt.val_bs = 1
    train_loader = DataLoader(train_dataset, batch_size=opt.bs, shuffle=True, num_workers=6)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.val_bs, shuffle=False, num_workers=3)

    
   
    print("dataset train=", len(train_dataset))
    print("dataset val=", len(valid_dataset))

    return train_loader, valid_loader

def prepare_test_data(opt, preprocessing_fn):

    test_df = df_from_csv_file_array(opt.test_CSVs)

    # test dataset without transformations for image visualization
    test_dataset = Dataset(
        test_df,
        grid_sizes=opt.grid_sizes_test,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=opt.classes,
        pyra = opt.pyra
    )

    print("Test dataset size=", len(test_dataset))

    return test_dataset
