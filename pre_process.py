import numpy as np
from torchvision import transforms
import os
from PIL import Image, ImageOps
import numbers
import torch
import random

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        th, tw = self.size
        return img.resize((tw, th))  # PIL uses (width, height) order

class RandomSizedCrop(object):
    """Crop the given tensor to random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: size of the crop
        scale: range of size of the cropped area (default: (0.08, 1.0))
        ratio: range of aspect ratio of the cropped area (default: (3/4, 4/3))
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be cropped.
        Returns:
            Tensor: Cropped image.
        """
        if len(img.shape) != 3:
            raise ValueError("Input tensor should have 3 dimensions (C, H, W)")
        
        _, height, width = img.shape
        area = height * width
        
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            
            w = int(round((target_area * aspect_ratio) ** 0.5))
            h = int(round((target_area / aspect_ratio) ** 0.5))
            
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                
                # Crop the tensor
                cropped = img[:, i:i+h, j:j+w]
                
                # Resize to target size using interpolation
                # Convert to PIL for resizing, then back to tensor
                from torchvision.transforms.functional import to_pil_image, to_tensor
                pil_img = to_pil_image(cropped)
                resized = pil_img.resize(self.size, Image.BILINEAR)
                return to_tensor(resized)
        
        # Fallback to center crop
        return self._center_crop(img)
    
    def _center_crop(self, img):
        _, height, width = img.shape
        crop_h, crop_w = self.size
        
        start_h = (height - crop_h) // 2
        start_w = (width - crop_w) // 2
        
        return img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respectively.
        std (sequence): Sequence of standard deviations for R, G, B channels respectively.
        meanfile (str): Path to .npy file containing mean values (alternative to mean)
    """

    def __init__(self, mean=None, std=None, meanfile=None):
        if mean is not None:
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std if std is not None else [1.0, 1.0, 1.0]).view(-1, 1, 1)
        elif meanfile is not None:
            arr = np.load(meanfile)
            # Convert to float32 and normalize to [0, 1] range, then reorder channels BGR -> RGB
            self.mean = torch.from_numpy(arr.astype('float32')/255.0)[[2,1,0],:,:]
            self.std = torch.ones_like(self.mean)
        else:
            raise ValueError("Either 'mean' or 'meanfile' must be provided")

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if self.mean.shape[0] != tensor.shape[0]:
            raise ValueError(f"Number of channels in mean ({self.mean.shape[0]}) doesn't match tensor ({tensor.shape[0]})")
        
        # Ensure mean and std are on the same device as tensor
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)
        
        return (tensor - mean) / std

class PlaceCrop(object):
    """Crops the given PIL.Image or tensor at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
        start_x (int): x coordinate of the top-left corner
        start_y (int): y coordinate of the top-left corner
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image or Tensor): Image to be cropped.
        Returns:
            PIL.Image or Tensor: Cropped image.
        """
        if isinstance(img, torch.Tensor):
            # Handle tensor input
            th, tw = self.size
            return img[:, self.start_y:self.start_y + th, self.start_x:self.start_x + tw]
        else:
            # Handle PIL Image input
            th, tw = self.size
            return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

class ForceFlip(object):
    """Horizontally flip the given PIL.Image or tensor."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image or Tensor): Image to be flipped.
        Returns:
            PIL.Image or Tensor: Flipped image.
        """
        if isinstance(img, torch.Tensor):
            # Handle tensor input - flip along width dimension
            return torch.flip(img, dims=[2])
        else:
            # Handle PIL Image input
            return img.transpose(Image.FLIP_LEFT_RIGHT)

class CenterCrop(object):
    """Crops the given PIL.Image or tensor at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image or Tensor): Image to be cropped.
        Returns:
            PIL.Image or Tensor: Cropped image.
        """
        if isinstance(img, torch.Tensor):
            # Handle tensor input (C, H, W)
            _, h, w = img.shape
            th, tw = self.size
            h_off = int((h - th) / 2.)
            w_off = int((w - tw) / 2.)
            return img[:, h_off:h_off+th, w_off:w_off+tw]
        else:
            # Handle PIL Image input
            w, h = img.size
            th, tw = self.size
            h_off = int((h - th) / 2.)
            w_off = int((w - tw) / 2.)
            return img.crop((w_off, h_off, w_off + tw, h_off + th))

def image_train(resize_size=256, crop_size=224, alexnet=False, grayscale=False):
    """Create training transform pipeline."""
    if grayscale:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(crop_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
        transforms.ToTensor(),
        normalize
    ])

def image_target(resize_size=256, crop_size=224, alexnet=False, grayscale=False):
    """Create target transform pipeline."""
    if grayscale:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False, grayscale=False):
    """Create test transform pipeline."""
    if grayscale:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
#audio preprocess
import torch
import torchaudio.transforms as T

class AudioTransform:
    def __init__(self, sample_rate=32000, n_mels=128, n_fft=2048, 
                 hop_length=512, normalize=True):
        self.sample_rate = sample_rate
        self.normalize = normalize
        
        # Optional: Add mel-spectrogram transform if needed
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
    def __call__(self, waveform):
        if self.normalize:
            # Normalize audio
            waveform = waveform / (torch.abs(waveform).max() + 1e-8)
        
        return waveform

def audio_train_transform(sample_rate=32000):
    """Training audio transforms with data augmentation"""
    return AudioTransform(sample_rate=sample_rate, normalize=True)

def audio_test_transform(sample_rate=32000):
    """Test audio transforms without augmentation"""
    return AudioTransform(sample_rate=sample_rate, normalize=True)