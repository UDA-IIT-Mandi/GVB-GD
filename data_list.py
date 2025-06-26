import os
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torchaudio

ImageFile.LOAD_TRUNCATED_IMAGES = True

def make_dataset(image_list, labels=None):
    if labels is not None:
        return [(image_list[i].strip(), labels[i, :]) for i in range(len(image_list))]
    else:
        if len(image_list[0].split()) > 2:
            return [(val.split()[0], np.array([int(x) for x in val.split()[1:]])) for val in image_list]
        else:
            return [(val.split()[0], int(val.split()[1])) for val in image_list]

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')  # Keep as single channel grayscale

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        self.imgs = make_dataset(image_list, labels)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images. Check paths or labels.")

        self.transform = transform
        self.target_transform = target_transform

        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        else:
            raise ValueError(f"Unsupported mode '{mode}', choose 'RGB' or 'L'")

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, loader=rgb_loader):
        self.imgs = make_dataset(image_list, labels)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images. Check paths or labels.")

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.values = [1.0] * len(self.imgs)  # Can be customized externally

    def set_values(self, values):
        if len(values) != len(self.imgs):
            raise ValueError("Length of values must match dataset size.")
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# Audio processing
class AudioList(Dataset):
    def __init__(self, audio_list, transform=None, sample_rate=32000, duration=10.0, mode='mono'):
        # Enhanced header filtering with multiple detection methods
        filtered_list = []
        skipped_headers = 0
        
        for i, line in enumerate(audio_list):
            line_clean = line.strip()
            
            # Skip empty lines
            if not line_clean:
                continue
            
            # Multiple header detection methods
            is_header = (
                line_clean.lower().startswith('filename') or
                'scene_label' in line_clean.lower() or
                'identifier' in line_clean.lower() or
                'source_label' in line_clean.lower() or
                line_clean == 'filename\tscene_label\tidentifier\tsource_label' or
                # Additional checks for exact header patterns
                line_clean.startswith('filename\t') or
                'filename' in line_clean and 'scene_label' in line_clean
            )
            
            if is_header:
                skipped_headers += 1
                print(f"Skipping header line {i}: {line_clean[:50]}...")
                continue
            
            # Validate it's a proper data line with tab separation
            if '\t' in line_clean:
                parts = line_clean.split('\t')
                if len(parts) >= 2:
                    filepath = parts[0]
                    # Must be an actual file path (contains / and ends with .wav)
                    if '/' in filepath and filepath.endswith('.wav') and os.path.exists(filepath):
                        filtered_list.append(line_clean)
                    elif '/' in filepath and filepath.endswith('.wav'):
                        # File path looks valid but doesn't exist - still include it for now
                        filtered_list.append(line_clean)
                    else:
                        print(f"Skipping invalid file path at line {i}: {filepath}")
                else:
                    print(f"Skipping malformed line {i}: {line_clean[:50]}...")
            else:
                print(f"Skipping non-tab-separated line {i}: {line_clean[:50]}...")
        
        self.audio_list = filtered_list
        self.transform = transform
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.mode = mode
        
        print(f"AudioList: Created with {len(self.audio_list)} samples")
        print(f"         Skipped {skipped_headers} header lines")
        print(f"         Filtered from {len(audio_list)} total lines")
        
        # Debug: show first few valid entries
        if len(self.audio_list) > 0:
            print("First 2 valid entries:")
            for i, entry in enumerate(self.audio_list[:2]):
                filepath = entry.split('\t')[0]
                print(f"  {i+1}: {filepath}")

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, index):
        line = self.audio_list[index].strip()
        
        # Additional safety check - should never happen with proper filtering
        if (line.lower().startswith('filename') or 'scene_label' in line.lower()):
            print(f"WARNING: Header found in filtered data at index {index}: {line}")
            # Return a safe default instead of crashing
            return torch.zeros(self.target_length), 0
        
        # Parse tab-separated values
        if '\t' in line:
            parts = line.split('\t')
            audio_path = parts[0]
            label = parts[1]  # This will be the scene label
        else:
            # Fallback for space-separated
            parts = line.split()
            audio_path = parts[0]
            label = parts[1] if len(parts) > 1 else 'unknown'
        
        # Convert scene label to integer if it's a string
        if isinstance(label, str):
            scene_map = {
                'airport': 0, 'bus': 1, 'metro': 2, 'metro_station': 3,
                'park': 4, 'public_square': 5, 'shopping_mall': 6,
                'street_pedestrian': 7, 'street_traffic': 8, 'tram': 9
            }
            label = scene_map.get(label, 0)
        else:
            label = int(label)
        
        # Load audio file
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return silence if file can't be loaded
            waveform = torch.zeros(1, self.target_length)
            sr = self.sample_rate
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad or truncate to target length
        current_length = waveform.shape[1]
        if current_length > self.target_length:
            # Random crop for training, center crop for testing
            if self.transform and 'train' in str(self.transform):
                start_idx = torch.randint(0, current_length - self.target_length + 1, (1,)).item()
                waveform = waveform[:, start_idx:start_idx + self.target_length]
            else:
                start_idx = (current_length - self.target_length) // 2
                waveform = waveform[:, start_idx:start_idx + self.target_length]
        elif current_length < self.target_length:
            padding = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Remove channel dimension for PaSST
        waveform = waveform.squeeze(0)
        
        # Apply audio transforms if specified
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, label

class DCASEDataset(Dataset):
    """Specific dataset class for DCASE audio files with proper header filtering"""
    def __init__(self, file_list, sample_rate=32000, duration=10.0, 
                 num_classes=10, transform=None):
        # Enhanced header filtering
        filtered_list = []
        skipped_headers = 0
        
        for i, line in enumerate(file_list):
            line_clean = line.strip()
            
            if not line_clean:
                continue
                
            # Multiple header detection methods
            is_header = (
                line_clean.lower().startswith('filename') or
                'scene_label' in line_clean.lower() or
                'identifier' in line_clean.lower() or
                'source_label' in line_clean.lower() or
                line_clean.startswith('filename\t') or
                'filename' in line_clean and 'scene_label' in line_clean
            )
            
            if is_header:
                skipped_headers += 1
                print(f"DCASEDataset: Skipping header line {i}: {line_clean[:50]}...")
                continue
            
            if '\t' in line_clean:
                parts = line_clean.split('\t')
                if len(parts) >= 2 and '/' in parts[0] and parts[0].endswith('.wav'):
                    filtered_list.append(line_clean)
        
        self.file_list = filtered_list
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.num_classes = num_classes
        self.transform = transform
        
        print(f"DCASEDataset: Created with {len(self.file_list)} samples")
        print(f"            Skipped {skipped_headers} header lines")
        print(f"            Filtered from {len(file_list)} total lines")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        line = self.file_list[idx].strip()
        
        # Additional safety check
        if (line.lower().startswith('filename') or 'scene_label' in line.lower()):
            print(f"WARNING: Header found in DCASEDataset at index {idx}: {line}")
            return torch.zeros(self.target_length), 0
        
        # Parse DCASE format: filename \t scene_label
        if '\t' in line:
            parts = line.split('\t')
            filename = parts[0]
            scene_label = parts[1]
        else:
            parts = line.split()
            filename = parts[0]
            scene_label = parts[1] if len(parts) > 1 else 'unknown'
        
        # Convert scene label to integer if it's a string
        if isinstance(scene_label, str):
            # Define DCASE scene mapping (adjust based on your dataset)
            scene_map = {
                'airport': 0, 'bus': 1, 'metro': 2, 'metro_station': 3,
                'park': 4, 'public_square': 5, 'shopping_mall': 6,
                'street_pedestrian': 7, 'street_traffic': 8, 'tram': 9
            }
            label = scene_map.get(scene_label, 0)
        else:
            label = int(scene_label)
        
        # Load and process audio
        try:
            waveform, sr = torchaudio.load(filename)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Adjust length
            if waveform.shape[1] > self.target_length:
                start_idx = (waveform.shape[1] - self.target_length) // 2
                waveform = waveform[:, start_idx:start_idx + self.target_length]
            elif waveform.shape[1] < self.target_length:
                padding = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            waveform = waveform.squeeze(0)  # Remove channel dimension
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            waveform = torch.zeros(self.target_length)
        
        return waveform, label