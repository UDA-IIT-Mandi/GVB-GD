import argparse
import os
import os.path as osp
from packaging import version
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
import random
import pdb
import math
import ipdb 
import torchaudio
from torch.utils.data import Dataset

from packaging import version

def debug_data_files(file_paths, num_lines=3):
    """Debug function to see what your data files look like"""
    for file_path in file_paths:
        print(f"\n=== DEBUG: {file_path} ===")
        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= num_lines:
                        break
                    line = line.strip()
                    print(f"Line {i}: '{line}'")
                    if '\t' in line:
                        parts = line.split('\t')
                        print(f"  Tab split: {len(parts)} parts: {parts}")
                    else:
                        parts = line.split()
                        print(f"  Space split: {len(parts)} parts: {parts}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        print("=" * 40)

# Audio Dataset for DCASE
class AudioList(Dataset):
    def __init__(self, audio_list, transform=None, sample_rate=32000, duration=10.0, mode='mono'):
        self.audio_list = audio_list
        self.transform = transform
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.mode = mode
        print(f"AudioList: Created with {len(audio_list)} samples")

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, index):
        line = self.audio_list[index].strip()
        
        # Robust parsing for different formats
        try:
            # Remove extra whitespace and handle multiple separators
            line = ' '.join(line.split())  # Normalize whitespace
            
            parts = line.split()
            if len(parts) >= 2:
                audio_path = parts[0]
                label = parts[1]
            elif len(parts) == 1:
                audio_path = parts[0] 
                label = "0"
            else:
                raise ValueError(f"Invalid line format: {line}")
            
            # Convert label to int
            try:
                label = int(label)
            except ValueError:
                # Handle text labels
                label_map = {
                    'airport': 0, 'bus': 1, 'metro': 2, 'metro_station': 3,
                    'park': 4, 'public_square': 5, 'shopping_mall': 6,
                    'street_pedestrian': 7, 'street_traffic': 8, 'tram': 9
                }
                label = label_map.get(str(label).lower(), 0)
                
        except Exception as e:
            print(f"Error parsing line {index}: '{line}' -> {e}")
            return torch.zeros(self.target_length), 0
        
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
            start_idx = torch.randint(0, current_length - self.target_length + 1, (1,)).item()
            waveform = waveform[:, start_idx:start_idx + self.target_length]
        elif current_length < self.target_length:
            padding = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Remove channel dimension for PaSST
        waveform = waveform.squeeze(0)
        
        # Apply audio transforms if specified
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, int(label)

# Audio transforms
class AudioTransform:
    def __init__(self, sample_rate=32000, normalize=True):
        self.sample_rate = sample_rate
        self.normalize = normalize
        
    def __call__(self, waveform):
        if self.normalize:
            # Normalize audio
            waveform = waveform / (torch.abs(waveform).max() + 1e-8)
        return waveform

def audio_train_transform(sample_rate=32000):
    return AudioTransform(sample_rate=sample_rate, normalize=True)

def audio_test_transform(sample_rate=32000):
    return AudioTransform(sample_rate=sample_rate, normalize=True)

def audio_classification_test(loader, model, gvbg=False):
    start_test = True
    with torch.no_grad():
        for i, data in enumerate(loader["test"]):
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            _, outputs, _ = model(inputs, gvbg=gvbg)
            
            if start_test:
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def image_classification_test(loader, model, gvbg=False, backbone='ResNet50'):
    if backbone == 'PaSST':
        return audio_classification_test(loader, model, gvbg)
    
    start_test = True
    with torch.no_grad():
       for i, data in enumerate(loader["test"]):
           inputs = data[0]
           labels = data[1]
           inputs = inputs.cuda()
           labels = labels.cuda()
           
           _, outputs, _ = model(inputs, gvbg=gvbg)
               
           if start_test:
               all_output = outputs.float()
               all_label = labels.float()
               start_test = False
           else:
               all_output = torch.cat((all_output, outputs.float()), 0)
               all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def correct_path(file_list):
    rm_string = '/DATA/disk1/hassassin/dataset/domain/OfficeHomeDataset/'
    for i in range(len(file_list)): 
        file_list[i] = file_list[i].replace(rm_string , "../data/office-home/")

    return file_list

def load_csv_skip_header(csv_path):
    """Load CSV file and skip the header row"""
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Skip first line if it looks like a header
    if lines and ('filename' in lines[0].lower() or 'scene_label' in lines[0].lower()):
        lines = lines[1:]  # Skip header
        print(f"Skipped header row in {csv_path}")
    
    # Remove empty lines and strip whitespace
    clean_lines = [line.strip() for line in lines if line.strip()]
    print(f"Loaded {len(clean_lines)} samples from {csv_path}")
    
    return clean_lines

def train(config):
    print("\n🚀 STARTING TRAINING FUNCTION")
    
    ## set pre-process
    prep_dict = {}
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    
    # Handle audio vs image preprocessing
    if config.get("backbone") == "PaSST":
        # Audio preprocessing
        sample_rate = config.get("sample_rate", 32000)
        prep_dict["source"] = audio_train_transform(sample_rate)
        prep_dict["target"] = audio_train_transform(sample_rate)
        prep_dict["test"] = audio_test_transform(sample_rate)
        print(f"✅ Using audio preprocessing with sample rate: {sample_rate}")
    else:
        # Image preprocessing (existing code)
        grayscale = config.get("grayscale", False)
        prep_params = config["prep"]['params'].copy()
        prep_params['grayscale'] = grayscale
        
        print(f"Grayscale mode: {grayscale}")
        print(f"Prep params: {prep_params}")
        
        prep_dict["source"] = prep.image_target(**prep_params)
        prep_dict["target"] = prep.image_target(**prep_params)
        prep_dict["test"] = prep.image_test(**prep_params)

    ## prepare data
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    
    print(f"📊 BATCH SIZES: train={train_bs}, test={test_bs}")
    
    # Load file lists
    source_list = load_csv_skip_header(data_config["source"]["list_path"])
    tgt_list = load_csv_skip_header(data_config["target"]["list_path"])
    test_list = load_csv_skip_header(data_config["test"]["list_path"])    
    
    # Handle path correction for office-home dataset only
    if config["dataset"] == "office-home":
        source_list = correct_path(source_list)
        tgt_list = correct_path(tgt_list)
        test_list = correct_path(test_list)

    # Create datasets based on backbone type
    if config.get("backbone") == "PaSST":
        # Audio datasets for DCASE
        sample_rate = config.get("sample_rate", 32000)
        duration = config.get("audio_length", 10.0)
        
        dsets["source"] = AudioList(source_list, 
                                   transform=prep_dict["source"],
                                   sample_rate=sample_rate,
                                   duration=duration)
        dsets["target"] = AudioList(tgt_list,
                                   transform=prep_dict["target"],
                                   sample_rate=sample_rate,
                                   duration=duration)
        dsets["test"] = AudioList(test_list,
                                 transform=prep_dict["test"],
                                 sample_rate=sample_rate,
                                 duration=duration)
        print(f"✅ Created audio datasets with {len(dsets['source'])} source, {len(dsets['target'])} target, {len(dsets['test'])} test samples")
    else:
        # Image datasets (existing code)
        mode = 'L' if config.get("grayscale", False) else 'RGB'
        print(f"Image loading mode: {mode}")
        
        dsets["source"] = ImageList(source_list, transform=prep_dict["source"], mode=mode)
        dsets["target"] = ImageList(tgt_list, transform=prep_dict["target"], mode=mode)
        dsets["test"] = ImageList(test_list, transform=prep_dict["test"], mode=mode)

    # Create data loaders
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs,
            shuffle=True, num_workers=4, drop_last=True, pin_memory=True, 
            persistent_workers=True if torch.cuda.is_available() else False)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs,
            shuffle=True, num_workers=4, drop_last=True, pin_memory=True,
            persistent_workers=True if torch.cuda.is_available() else False)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs,
                                shuffle=False, num_workers=4, pin_memory=True,
                                persistent_workers=True if torch.cuda.is_available() else False)

    print("✅ DataLoaders created successfully")

    ## set base network
    class_num = config["network"]["params"]["class_num"]
    net_config = config["network"]
    
    print(f"\n🔧 NETWORK SETUP:")
    print(f"   Network name: {net_config['name']}")
    print(f"   Network params: {net_config['params']}")
    print(f"   Class num: {class_num}")
    
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    
    print(f"✅ Base network created and moved to CUDA")

    ## add additional network for some methods
    print(f"\n🎯 ADVERSARIAL NETWORK SETUP:")
    print(f"   Input dimension: {class_num}")
    print(f"   Hidden dimension: 1024")
    
    ad_net = network.AdversarialNetwork(768, 1024)
    ad_net = ad_net.cuda()
    
    print(f"✅ Adversarial network created")
    print(f"   ad_net.ad_layer1.in_features: {ad_net.ad_layer1.in_features}")
    print(f"   ad_net.ad_layer1.out_features: {ad_net.ad_layer1.out_features}")
 
    ## set optimizer
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list,
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    print(f"\n⚙️ OPTIMIZER SETUP:")
    print(f"   Type: {optimizer_config['type']}")
    print(f"   Params: {optimizer_config['optim_params']}")
    print(f"   Learning rates: {param_lr}")

    # multi gpu - Updated for modern PyTorch
    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        device_ids = [int(gpu_id) for gpu_id in gpus]
        ad_net = nn.DataParallel(ad_net, device_ids=device_ids)
        base_network = nn.DataParallel(base_network, device_ids=device_ids)
        print(f"🔀 Multi-GPU setup with devices: {device_ids}")
    
    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    
    print(f"\n📈 TRAINING SETUP:")
    print(f"   Source batches per epoch: {len_train_source}")
    print(f"   Target batches per epoch: {len_train_target}")
    print(f"   Total iterations: {config['num_iterations']}")
    print(f"   Test interval: {config['test_interval']}")
    print(f"   Print interval: {config['print_num']}")
    
    # Create iterators outside the loop for better performance
    iter_source = iter(dset_loaders["source"])
    iter_target = iter(dset_loaders["target"])
    
    print(f"\n🚀 STARTING TRAINING LOOP\n")
    
    for i in range(config["num_iterations"]):
        
        # Detailed debug for first 5 iterations and every 100th
        debug_this_iter = (i < 5) or (i % 100 == 0)
        
        if debug_this_iter:
            print(f"\n{'='*60}")
            print(f"🔍 ITERATION {i} DEBUG START")
            print(f"{'='*60}")
        
        # test
        if i % config["test_interval"] == config["test_interval"] - 1:
            print(f"\n🧪 TESTING AT ITERATION {i}")
            base_network.eval()
            temp_acc = image_classification_test(dset_loaders, base_network, 
                                               gvbg=config["GVBG"], 
                                               backbone=config.get("backbone", "ResNet50"))
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
                print(f"🎉 NEW BEST ACCURACY: {best_acc:.5f}")
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)
            
        # save model
        if i % config["snapshot_interval"] == 0:
            print(f"💾 SAVING MODEL AT ITERATION {i}")
            torch.save(base_network.state_dict(), osp.join(config["output_path"],
                "iter_{:05d}_model.pth.tar".format(i)))

        ## train one iter
        base_network.train()
        ad_net.train()
        loss_params = config["loss"]                  
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if debug_this_iter:
            print(f"📋 Training mode set, optimizer zeroed")

        # dataloader - Recreate iterators when exhausted
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
            if debug_this_iter:
                print(f"🔄 Recreated source iterator (iteration {i})")
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
            if debug_this_iter:
                print(f"🔄 Recreated target iterator (iteration {i})")
            
        # network
        try:
            inputs_source, labels_source = next(iter_source)
            inputs_target, _ = next(iter_target)
        except StopIteration:
            # Handle iterator exhaustion gracefully
            if debug_this_iter:
                print(f"⚠️ StopIteration caught, recreating iterators")
            iter_source = iter(dset_loaders["source"])
            iter_target = iter(dset_loaders["target"])
            inputs_source, labels_source = next(iter_source)
            inputs_target, _ = next(iter_target)
            
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        if debug_this_iter:
            print(f"📥 DATA LOADED:")
            print(f"   inputs_source shape: {inputs_source.shape}")
            print(f"   inputs_target shape: {inputs_target.shape}")
            print(f"   labels_source shape: {labels_source.shape}")
            print(f"   labels_source unique: {torch.unique(labels_source).tolist()}")

        # Forward pass (same for both PaSST and ResNet)
        if debug_this_iter:
            print(f"🔀 FORWARD PASS - SOURCE:")
            
        features_source, outputs_source, focal_source = base_network(inputs_source, gvbg=config["GVBG"])
        
        if debug_this_iter:
            print(f"   features_source shape: {features_source.shape}")
            print(f"   outputs_source shape: {outputs_source.shape}")
            print(f"   focal_source shape: {focal_source.shape}")
            print(f"   outputs_source stats: min={outputs_source.min():.4f}, max={outputs_source.max():.4f}, mean={outputs_source.mean():.4f}")
            print(f"🔀 FORWARD PASS - TARGET:")
            
        features_target, outputs_target, focal_target = base_network(inputs_target, gvbg=config["GVBG"])
        
        if debug_this_iter:
            print(f"   features_target shape: {features_target.shape}")
            print(f"   outputs_target shape: {outputs_target.shape}")
            print(f"   focal_target shape: {focal_target.shape}")
            print(f"   outputs_target stats: min={outputs_target.min():.4f}, max={outputs_target.max():.4f}, mean={outputs_target.mean():.4f}")
            
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0) 
        focals = torch.cat((focal_source, focal_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        if debug_this_iter:
            print(f"🔗 CONCATENATED TENSORS:")
            print(f"   features shape: {features.shape}")
            print(f"   outputs shape: {outputs.shape}")
            print(f"   focals shape: {focals.shape}")
            print(f"   features_out shape: {features.shape}")
            print(f"   features_out stats: min={features.min():.6f}, max={features.max():.6f}, mean={features.mean():.6f}")
            print(f"   features_out sum per sample: {features.sum(dim=1)[:3].tolist()} (should be ~1.0)")

        # Coefficient calculation
        coeff = network.calc_coeff(i)
        
        if debug_this_iter:
            print(f"📊 LOSS CALCULATION PREP:")
            print(f"   calc_coeff({i}): {coeff:.8f}")
            print(f"   GVBD parameter: {config['GVBD']}")
            print(f"   trade_off parameter: {loss_params['trade_off']}")

        # Test adversarial network before loss calculation
        if debug_this_iter:
            print(f"🎯 TESTING ADVERSARIAL NETWORK:")
            try:
                with torch.enable_grad():
                    test_softmax = features.clone().detach().requires_grad_(True)
                    print(f"   test_softmax requires_grad: {test_softmax.requires_grad}")
                    test_ad_out = ad_net(test_softmax)
                    print(f"   ad_net output shape: {test_ad_out.shape}")
                    print(f"   ad_net output stats: min={test_ad_out.min():.6f}, max={test_ad_out.max():.6f}, mean={test_ad_out.mean():.6f}")
                    
                    # Check if outputs are constant
                    ad_std = test_ad_out.std()
                    print(f"   ad_net output std: {ad_std:.8f} (should be > 0.001 for learning)")
                    if ad_std < 0.001:
                        print(f"   ⚠️ WARNING: Ad network outputs nearly constant values!")
                        
            except Exception as e:
                print(f"   ❌ ERROR testing adversarial network: {e}")

        # loss calculation
        if debug_this_iter:
            print(f"🔥 CALLING LOSS.GVB FUNCTION:")
            print(f"   Input 1 (features): shape={features.shape}, dtype={features.dtype}")
            print(f"   Input 2 (focals): shape={focals.shape}, dtype={focals.dtype}")
            print(f"   Input 3 (ad_net): type={type(ad_net)}")
            print(f"   Input 4 (coeff): {coeff}")
            print(f"   Input 5 (GVBD): {config['GVBD']}")
            
        transfer_loss, mean_entropy, gvbg, gvbd = loss.GVB([softmax_out, focals,features], ad_net, coeff, GVBD=config['GVBD'])
        
        if debug_this_iter:
            print(f"✅ LOSS.GVB OUTPUTS:")
            print(f"   transfer_loss: {transfer_loss.item():.8f}")
            print(f"   mean_entropy: {mean_entropy.item():.8f}")
            print(f"   gvbg: {gvbg.item():.8f}")
            print(f"   gvbd: {gvbd.item():.8f}")
            
            # Check if transfer loss is exactly ln(2)
            ln2 = np.log(2)
            if abs(transfer_loss.item() - ln2) < 1e-6:
                print(f"   ⚠️ WARNING: Transfer loss = ln(2) = {ln2:.8f} (possible constant output)")
        
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        
        if debug_this_iter:
            print(f"📊 CLASSIFIER LOSS:")
            print(f"   classifier_loss: {classifier_loss.item():.8f}")
            
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + config["GVBG"] * gvbg + abs(config['GVBD']) * gvbd

        if debug_this_iter:
            print(f"📊 TOTAL LOSS CALCULATION:")
            print(f"   {loss_params['trade_off']:.3f} * {transfer_loss.item():.6f} + {classifier_loss.item():.6f} + {config['GVBG']:.3f} * {gvbg.item():.6f} + {abs(config['GVBD']):.3f} * {gvbd.item():.6f}")
            print(f"   = {loss_params['trade_off'] * transfer_loss.item():.6f} + {classifier_loss.item():.6f} + {config['GVBG'] * gvbg.item():.6f} + {abs(config['GVBD']) * gvbd.item():.6f}")
            print(f"   = {total_loss.item():.8f}")

        if i % config["print_num"] == 0:
            log_str = "iter: {:05d}, transferloss: {:.5f}, classifier_loss: {:.5f}, mean entropy:{:.5f}, gvbg:{:.5f}, gvbd:{:.5f}".format(i, transfer_loss, classifier_loss, mean_entropy, gvbg, gvbd)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)

        if debug_this_iter:
            print(f"🔄 BACKWARD PASS:")
            
        total_loss.backward()############################
        
        if debug_this_iter:
            print(f"   Backward pass completed")
            # Check gradients
            base_grad_norm = torch.nn.utils.clip_grad_norm_(base_network.parameters(), float('inf'))
            ad_grad_norm = torch.nn.utils.clip_grad_norm_(ad_net.parameters(), float('inf'))
            print(f"   Base network grad norm: {base_grad_norm:.8f}")
            print(f"   Adversarial network grad norm: {ad_grad_norm:.8f}")
            
            if base_grad_norm < 1e-8:
                print(f"   ⚠️ WARNING: Base network gradients very small!")
            if ad_grad_norm < 1e-8:
                print(f"   ⚠️ WARNING: Adversarial network gradients very small!")
        torch.nn.utils.clip_grad_norm_(base_network.parameters(), max_norm=1.0)#to prevent nan clapping
        torch.nn.utils.clip_grad_norm_(ad_net.parameters(), max_norm=1.0)      #to prevent nan clapping  
        optimizer.step()
        
        if debug_this_iter:
            print(f"   Optimizer step completed")
            print(f"🔍 ITERATION {i} DEBUG END")
            print(f"{'='*60}\n")
        
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"📊 Best accuracy achieved: {best_acc:.5f}")
    return best_acc

if __name__ == "__main__":

    assert version.parse(torch.__version__) >= version.parse('2.0.0'), 'PyTorch>=2.0.0 is required'

    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet50")
    parser.add_argument('--dset', type=str, default='office-home', help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='data/Art.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='data/Clipart.txt', help="The target dataset path list")
    parser.add_argument('--test_dset_path', type=str, default=None, help="Test dataset path ")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--print_num', type=int, default=100, help="interval of two print loss")
    parser.add_argument('--num_iterations', type=int, default=10002, help="interation num ")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--trade_off', type=float, default=1, help="parameter for transfer loss")
    parser.add_argument('--batch_size', type=int, default=36, help="batch size")
    parser.add_argument('--GVBG', type=float, default=0, help="lambda: parameter for GVBG (if lambda==0 then GVBG is not utilized)")
    parser.add_argument('--GVBD', type=float, default=0, help="mu: parameter for GVBD (if mu==0 then GVBD is not utilized)")
    parser.add_argument('--CDAN', type=bool, default=False, help="utilize CDAN or not)")
    parser.add_argument('--grayscale', action='store_true', help="Convert images to grayscale")
    parser.add_argument('--backbone', type=str, default='ResNet50', help="Options: ResNet50, PaSST")
    parser.add_argument('--sample_rate', type=int, default=32000, help="Audio sample rate")
    parser.add_argument('--audio_length', type=float, default=10.0, help="Audio clip length in seconds")
    args = parser.parse_args()
    
    # Debug data files first
    print("=== DEBUGGING DATA FILES ===")
    debug_data_files([args.s_dset_path, args.t_dset_path])
    print("=== END DEBUG ===")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Initialize config dictionary FIRST
    config = {}
    config["GVBG"] = args.GVBG
    config["GVBD"] = args.GVBD 
    config["CDAN"] = args.CDAN
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iterations 
    config["print_num"] = args.print_num
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = args.dset + "/" + args.output_dir
    config["backbone"] = args.backbone
    config["sample_rate"] = args.sample_rate
    config["audio_length"] = args.audio_length

    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"], exist_ok=True)
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    # Initialize basic config
    config["prep"] = {'params':{"resize_size":224, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":args.trade_off}
    config["grayscale"] = args.grayscale

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9,
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv",
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":args.batch_size},
                      "target":{"list_path":args.t_dset_path, "batch_size":args.batch_size},
                      "test": {"list_path": args.test_dset_path, "batch_size": args.batch_size}}

    # Dataset-specific configuration
    if config["dataset"] == "office-home":
        seed = 2019
        config["optimizer"]["lr_param"]["lr"] = 0.0001
        config["network"] = {"params":{"class_num": 65}}
    elif config["dataset"] == "office":
        seed = 2019
        if   ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
             config["optimizer"]["lr_param"]["lr"] = 0.001
        elif ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
             ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
             config["optimizer"]["lr_param"]["lr"] = 0.0003
        config["network"] = {"params":{"class_num": 31}}
    elif config["dataset"] == "visda":
        seed = 9297
        config["optimizer"]["lr_param"]["lr"] = 0.0003
        config["network"] = {"params":{"class_num": 12}}
    elif config["dataset"] == "mnist-svhn":
        seed = 2025
        config["optimizer"]["lr_param"]["lr"] = 0.0001
        config["network"] = {"params":{"class_num": 10}}
        config["grayscale"] = True
        print("MNIST-SVHN detected: Enabling grayscale conversion")
        config["prep"] = {'params':{"resize_size":28, "crop_size":28, 'alexnet':False}}
    elif config["dataset"] == "dcase":
        seed = 2020
        config["optimizer"]["lr_param"]["lr"] = 0.0001
        config["network"] = {"params": {"class_num": 10}}
        print("DCASE dataset detected: Configuring for acoustic scene classification")
    elif config["dataset"] == "dcase-task1a":
        seed = 2020
        config["optimizer"]["lr_param"]["lr"] = 0.0001
        config["network"] = {"params": {"class_num": 10}}
        print("DCASE Task 1A detected: Acoustic Scene Classification")
    elif config["dataset"] == "dcase-task1b":
        seed = 2020
        config["optimizer"]["lr_param"]["lr"] = 0.0001
        config["network"] = {"params": {"class_num": 3}}  # Device A, B, C
        print("DCASE Task 1B detected: Acoustic Scene Classification with device mismatch")
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    
    # Network configuration based on backbone
    if args.backbone == 'PaSST':
        config["network"]["name"] = network.PaSSTFc
        config["network"]["params"].update({
            "use_bottleneck": False, 
            "bottleneck_dim": 224, 
            "new_cls": True,
            "device": f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        })
        print(f"Using PaSST backbone with sample rate: {args.sample_rate}")
        
    elif "ResNet" in args.net or args.backbone.startswith('ResNet'):
        input_channels = 1 if config.get("grayscale", False) else 3
        print(f"DEBUG: Setting input_channels to {input_channels}")
        print(f"DEBUG: config['grayscale'] = {config.get('grayscale', False)}")
        
        config["network"]["name"] = network.ResNetFc
        config["network"]["params"].update({
            "resnet_name": args.net, 
            "use_bottleneck": False, 
            "bottleneck_dim": 224, 
            "new_cls": True,
            "input_channels": input_channels
        })
    else:
        raise ValueError(f'Backbone {args.backbone} not recognized. Please define your own backbone here.')
    
    # Set seed
    seed = globals().get('seed', 2019)
    print(f"Using seed: {seed}")
    print(f"Using backbone: {args.backbone}")
    print(f"Dataset: {config['dataset']}")
    print(f"Number of classes: {config['network']['params']['class_num']}")
    
    # Improved random seed setting for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Updated for modern PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    # Set generator for DataLoader reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)