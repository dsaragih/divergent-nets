
import pandas as pd
import albumentations as albu
import cv2
from torch.utils.data import DataLoader
import os
import numpy as np
import pickle
from data.dataset import Dataset

def df_from_csv_file_array(csv_file_arrya):

    df =pd.DataFrame(columns=["image_path", "mask_path"])

    for csv in csv_file_arrya:
        temp_df = pd.read_csv(csv)

        df = pd.concat([df, temp_df], ignore_index=True)

    return df

def df_from_img_dir(img_dir_path, n=None):
    # img_dir_path contains two directories: images and masks
    # each directory contains images with the same name
    # e.g. img_dir_path/images/1.png and img_dir_path/masks/1.png
    # will be paired together in the dataframe

    df = pd.DataFrame(columns=["image_path", "mask_path"])
    mask_path = os.path.join(img_dir_path, "masks")
    dir_list = list(sorted(os.listdir(os.path.join(img_dir_path, "masked-images"))))
    if n is not None:
        dir_list = dir_list[:n]
    for img in dir_list:
        new_row = pd.Series({"image_path": os.path.join(img_dir_path, "masked-images", img), "mask_path": os.path.join(mask_path, img)})
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    
    return df

def _load_all_pkl_images(pkl_path, n_samples=1):
    with open(pkl_path, "rb") as f:
        images = pickle.load(f)
    """
    images is a dictionary: {img_path: generated images}
    generated images has shape (N, 256, 256, 4)
    """
    first = list(images.values())[0]
    idx = np.random.randint(0, first.shape[0], size=n_samples)
    paths = []
    images_arr = []
    for path, imgs in images.items():
        choice = imgs[idx]
        if len(choice.shape) == 3:
            choice = np.expand_dims(choice, axis=0)
        paths.extend([path] * len(choice))
        images_arr.append(choice)
    images = np.concatenate(images_arr, axis=0)
    # If mean < 128, multiply by 255
    images_mean = np.mean(images)
    if images_mean < 1: # heuristic to determine if images are in [0, 1] or [0, 255]
        images *= 255
    return images

def _load_given_pkl_image(pkl_path, img_path, images=None):
    """
    Given original image path, load the generated images from the pkl file and return the corresponding batch of images
    """
    if images is None:
        with open(pkl_path, "rb") as f:
            images = pickle.load(f)

    pkl_path_dir = os.path.dirname(pkl_path)
    # Return key with matching basename
    img_path = os.path.basename(img_path)
    ims = None

    # This makes sure that the images do not conflict with each other
    idx = 0
    for path, imgs in images.items():
        # img_path is the basename of the path
        if img_path == os.path.basename(path):
            # Set idx to the index of path in paths
            idx = list(images.keys()).index(path) * imgs.shape[0]
            ims = imgs
            break
    if ims is None:
        raise ValueError("Image not found in pickle file.")
    
    # Paths like (i.jpg, i+1.jpg, ..., i+n-1.jpg), n = imgs.shape[0]
    img_paths = []
    mask_paths = []
    if np.mean(ims) < 1: # heuristic to determine if images are in [0, 1] or [0, 255]
        ims *= 255
    for i, im in enumerate(ims):
        img = im[:, :, :3]
        mask = im[:, :, 3]
        mask = np.where(mask > 128, 255, 0).astype(np.uint8)
        img_path = os.path.join(pkl_path_dir, "masked-images", str(idx+i) + ".jpg")
        mask_path = os.path.join(pkl_path_dir, "masks", str(idx+i) + ".jpg")
        # cv2 expects BGR
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # mask is one channel
        cv2.imwrite(mask_path, mask)
        img_paths.append(img_path)
        mask_paths.append(mask_path)
    return img_paths, mask_paths
    

def augment_train_df(df, pkl_path, n_samples=1):
    """
    For each image in df, there's a 50% chance it will be replaced with {n_samples} augmented images.
    """
    augmented_df = pd.DataFrame(columns=["image_path", "mask_path"])
    aug_counter = 0
    with open(pkl_path, "rb") as f:
        images = pickle.load(f)
    for i, row in df.iterrows():
        # 50% chance to augment
        if np.random.random() < 0.5:
            img_paths, mask_paths = _load_given_pkl_image(pkl_path, row["image_path"], images)
            print(f"Augmenting image {os.path.basename(row['image_path'])}")
            print(f"Replacing with {n_samples} augmented images, starting at {os.path.basename(img_paths[0])}")

            assert n_samples <= len(img_paths)
            rng = np.random.default_rng()
            iters = rng.choice(len(img_paths), size=n_samples, replace=False)
            aug_counter += 1
            for i in iters:
                new_row = pd.Series({"image_path": img_paths[i], "mask_path": mask_paths[i]})
                augmented_df = pd.concat([augmented_df, new_row.to_frame().T], ignore_index=True)
                
        new_row = pd.Series({"image_path": row["image_path"], "mask_path": row["mask_path"]})
        augmented_df = pd.concat([augmented_df, new_row.to_frame().T], ignore_index=True)

    print(f"Augmented {aug_counter * n_samples} images.")
    return augmented_df

    

def df_from_pkl(pkl_path, n_samples=1):
    """pkl_path contains a numpy array of shape (N, H, W, 4), where
    the last dimension is the mask. The first three dimensions are the image.

    NOTE: idx may be an array of indices, in which case the function will return a dataframe where, starting from the first index in idx, {len(idx)} consecutive indices are samples from the same image.

    Output .jpg images to a directory and return a dataframe with the paths to the images and masks.
    """
    pkl_path_dir = os.path.dirname(pkl_path)
    df = pd.DataFrame(columns=["image_path", "mask_path"])
    images = _load_all_pkl_images(pkl_path, n_samples)

    for i, im in enumerate(images):
        img = im[:, :, :3]
        mask = im[:, :, 3]
        mask = np.where(mask > 128, 255, 0).astype(np.uint8)
        img_path = os.path.join(pkl_path_dir, "masked-images", str(i) + ".jpg")
        mask_path = os.path.join(pkl_path_dir, "masks", str(i) + ".jpg")
        # cv2 expects BGR
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # mask is one channel
        cv2.imwrite(mask_path, mask)
        new_row = pd.Series({"image_path": img_path, "mask_path": mask_path})
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    return df


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        albu.Resize(height=256, width=256, always_apply=True),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        )
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
    """Prepare data for training, testing, and validation.
    """
    np.random.seed(0)
    N_DATA = 16
    if opt.mode == "aug_syn_train":
        train_val_df = df_from_img_dir(opt.img_dir)
        # 80, 20 split
        train_df = train_val_df.sample(frac=0.8, random_state=0)
        val_df = train_val_df.drop(train_df.index)
        # Take small sample for training
        train_df = train_df.sample(n=N_DATA, random_state=0)
        train_df = augment_train_df(train_df, opt.pkl_path, n_samples=opt.n_samples)
    elif opt.mode == "full_syn_train":
        train_val_df = df_from_pkl(opt.pkl_path, n_samples=opt.n_samples)
        train_df = train_val_df.sample(frac=0.8, random_state=0)
        val_df = train_val_df.drop(train_df.index)
        train_df = train_df.sample(n=N_DATA, random_state=0)
    elif opt.mode == "app_syn_train":
        train_val_df = df_from_pkl(opt.pkl_path, n_samples=opt.n_samples)
        # Append real images
        real_df = df_from_img_dir(opt.img_dir)
        train_val_df = pd.concat([train_val_df, real_df], ignore_index=True)
        train_df = train_val_df.sample(frac=0.8, random_state=0)
        val_df = train_val_df.drop(train_df.index)
    elif opt.mode == "real_train":
        train_val_df = df_from_img_dir(opt.img_dir)
        # 80, 20 split
        train_df = train_val_df.sample(frac=0.8, random_state=0)
        val_df = train_val_df.drop(train_df.index)
        # Take small sample for training
        train_df = train_df.sample(n=N_DATA, random_state=0)
        
    len_val = len(val_df)
    test_dataset = prepare_test_data(opt, preprocessing_fn, len_val)
    # train_df = df_from_csv_file_array(opt.train_CSVs)
    # val_df = df_from_csv_file_array(opt.val_CSVs)

    # Set augmentation = get_validation_augmentation() for no data augmentation
    train_dataset = Dataset(
        train_df,
        grid_sizes=opt.grid_sizes_train,
        augmentation=get_validation_augmentation(), 
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
    print("Test dataset size=", len(test_dataset))
    print("train_loader shape:", next(iter(train_loader))[0].shape)
    return train_loader, valid_loader, test_dataset

def prepare_test_data(opt, preprocessing_fn, n):

    test_df = df_from_img_dir(opt.test_dir, 200)

    # test dataset without transformations for image visualization
    test_dataset = Dataset(
        test_df,
        grid_sizes=opt.grid_sizes_test,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=opt.classes,
        pyra = opt.pyra
    )

    return test_dataset
