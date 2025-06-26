import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import math

# Add PaSST imports
try:
    from hear21passt.base import get_basic_model
    from hear21passt.models.passt import PaSST
    PASST_AVAILABLE = True
except ImportError:
    print("Warning: PaSST not installed. Install with: pip install hear21passt")
    PASST_AVAILABLE = False
    
    def get_basic_model(mode="embed_only"):
        raise NotImplementedError("PaSST not available. Install with: pip install hear21passt")

# --- Fixed np.float deprecation ---
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=1500.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

# --- Initialization ---
def init_weights(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

resnet_dict = {
    "ResNet18": models.resnet18,
    "ResNet34": models.resnet34,
    "ResNet50": models.resnet50,
    "ResNet101": models.resnet101,
    "ResNet152": models.resnet152
}

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

# --- Main ResNet Model ---
class ResNetFc(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000, input_channels=3):
        super(ResNetFc, self).__init__()
        
        print(f"ResNetFc initialized with input_channels={input_channels}")
        
        model_resnet = resnet_dict[resnet_name](weights='DEFAULT')
        
        print(f"Original conv1 shape: {model_resnet.conv1.weight.shape}")
        
        # Modify first conv layer if input_channels != 3
        if input_channels != 3:
            print(f"Modifying conv1 for {input_channels} input channels")
            original_conv1 = model_resnet.conv1
            model_resnet.conv1 = nn.Conv2d(
                input_channels, 
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias is not None
            )
            
            print(f"New conv1 shape: {model_resnet.conv1.weight.shape}")
            
            # Initialize new conv layer weights
            if input_channels == 1:
                # For grayscale, average the RGB weights
                with torch.no_grad():
                    model_resnet.conv1.weight[:, 0, :, :] = original_conv1.weight.mean(dim=1)
                    if original_conv1.bias is not None and model_resnet.conv1.bias is not None:
                        model_resnet.conv1.bias.copy_(original_conv1.bias)
                print("Initialized conv1 weights for grayscale input")
        else:
            print("Using original 3-channel conv1")
                        
        self.feature_layers = nn.Sequential(
            model_resnet.conv1,
            model_resnet.bn1,
            model_resnet.relu,
            model_resnet.maxpool,
            model_resnet.layer1,
            model_resnet.layer2,
            model_resnet.layer3,
            model_resnet.layer4,
            model_resnet.avgpool
        )

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        self.sigmoid = nn.Sigmoid()

        if new_cls:
            if use_bottleneck:
                self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.gvbg = nn.Linear(bottleneck_dim, class_num)
                self.focal1 = nn.Linear(class_num, class_num)
                self.focal2 = nn.Linear(class_num, 1)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.gvbg.apply(init_weights)
                self.focal1.apply(init_weights)
                self.focal2.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.gvbg = nn.Linear(model_resnet.fc.in_features, class_num)
                self.focal1 = nn.Linear(class_num, class_num)
                self.focal2 = nn.Linear(class_num, 1)
                self.fc.apply(init_weights)
                self.gvbg.apply(init_weights)
                self.focal1.apply(init_weights)
                self.focal2.apply(init_weights)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x, gvbg=True):
        x = self.feature_layers(x)
        x = torch.flatten(x, start_dim=1)  # safer than .view(...)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        
        y = self.fc(x)
        
        if self.new_cls:
            if gvbg:
                bridge = self.gvbg(x)
                focal = self.focal2(torch.relu(self.focal1(y)))
                return x, y, focal
            else:
                focal = torch.zeros(y.size(0), 1, device=y.device)
                return x, y, focal
        else:
            focal = torch.zeros(y.size(0), 1, device=y.device)
            return x, y, focal

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        params = []
        if self.new_cls:
            params.append({"params": self.feature_layers.parameters(), "lr_mult": 1, "decay_mult": 2})
            if self.use_bottleneck:
                params.append({"params": self.bottleneck.parameters(), "lr_mult": 10, "decay_mult": 2})
            params.append({"params": self.fc.parameters(), "lr_mult": 10, "decay_mult": 2})
            if hasattr(self, 'gvbg'):
                params.append({"params": self.gvbg.parameters(), "lr_mult": 10, "decay_mult": 2})
            if hasattr(self, 'focal1'):
                params.append({"params": self.focal1.parameters(), "lr_mult": 10, "decay_mult": 2})
            if hasattr(self, 'focal2'):
                params.append({"params": self.focal2.parameters(), "lr_mult": 10, "decay_mult": 2})
        else:
            params.append({"params": self.parameters(), "lr_mult": 1, "decay_mult": 2})
        return params

# --- Enhanced PaSST Model with Native 768-dim Embeddings ---
class PaSSTFc(torch.nn.Module):
    def __init__(self, device=None, class_num=10, use_bottleneck=False, bottleneck_dim=256, new_cls=True):
        super(PaSSTFc, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_num = class_num
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        
        if not PASST_AVAILABLE:
            raise ImportError("PaSST not available. Install with: pip install hear21passt")
            
        # Use embedding mode to get native 768-dim embeddings
        from hear21passt.base import get_basic_model
        self.passt_wrapper = get_basic_model(mode="embed_only")
        self.passt_wrapper.to(self.device)
        print("✅ Using PaSST native 768-dim embeddings directly (no projection needed)")
        
        # Work directly with 768-dim embeddings
        feature_dim = 768
        
        # Classification layers
        if use_bottleneck and new_cls:
            self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.gvbg = nn.Linear(bottleneck_dim, class_num)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(feature_dim, class_num)
            self.gvbg = nn.Linear(feature_dim, class_num)
            self.__in_features = feature_dim
            
        # Focal layers for GVB
        if new_cls:
            self.focal1 = nn.Linear(class_num, class_num)
            self.focal2 = nn.Linear(class_num, 1)
            
        # Initialize weights for new layers
        for module in [self.fc, self.gvbg, getattr(self, 'bottleneck', None), 
                      getattr(self, 'focal1', None), getattr(self, 'focal2', None)]:
            if module is not None:
                module.apply(init_weights)
        
        print(f"✅ PaSST model initialized:")
        print(f"   - Feature dimension: {feature_dim}")
        print(f"   - Use bottleneck: {use_bottleneck}")
        print(f"   - Bottleneck dim: {bottleneck_dim if use_bottleneck else 'N/A'}")
        print(f"   - Output features: {self.__in_features}")
        print(f"   - Classes: {class_num}")

    def forward(self, audio_waveform, gvbg=True):
        # Ensure input is on correct device
        audio_waveform = audio_waveform.to(self.device)
        
        # Get native 768-dim embeddings directly from PaSST
        embeddings = self.passt_wrapper(audio_waveform)  # [batch_size, 768]
        
        # Optional: Add L2 normalization for better stability
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Apply bottleneck if configured
        x = embeddings
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        
        # Classification
        outputs = self.fc(x)
        
        # Focal computation for GVB
        if self.new_cls:
            focal = self.focal2(torch.relu(self.focal1(outputs)))
        else:
            focal = torch.zeros(outputs.size(0), 1, device=outputs.device)
        
        # Return embeddings (768-dim), predictions, and focal values
        return embeddings, outputs, focal
    
    def get_parameters(self):
        params = []
        # Pre-trained PaSST with lower learning rate
        params.append({"params": self.passt_wrapper.parameters(), "lr_mult": 0.1, "decay_mult": 2})
        
        # Classification layers with higher learning rate
        if self.use_bottleneck and self.new_cls:
            params.append({"params": self.bottleneck.parameters(), "lr_mult": 10, "decay_mult": 2})
        params.append({"params": self.fc.parameters(), "lr_mult": 10, "decay_mult": 2})
        params.append({"params": self.gvbg.parameters(), "lr_mult": 10, "decay_mult": 2})
        
        if self.new_cls:
            params.append({"params": self.focal1.parameters(), "lr_mult": 10, "decay_mult": 2})
            params.append({"params": self.focal2.parameters(), "lr_mult": 10, "decay_mult": 2})
            
        return params
    
    def output_num(self):
        return self.__in_features

# --- FIXED Adversarial Network (Removed BatchNorm) ---
class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, dual_output=False):
        super(AdversarialNetwork, self).__init__()
        self.dual_output = dual_output
        
        # REMOVED: self.feature_norm = nn.BatchNorm1d(in_feature)
        # BatchNorm can cause gradient flow issues in adversarial training
        
        # Adversarial layers
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        
        # Additional layer for dual output mode (GVBD)
        if dual_output:
            self.fc_layer = nn.Linear(hidden_size, 1)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self.apply(init_weights)
        
        # GRL parameters
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0
        
        print(f"✅ Adversarial Network initialized (NO BatchNorm):")
        print(f"   - Input features: {in_feature}")
        print(f"   - Hidden size: {hidden_size}")
        print(f"   - Dual output: {dual_output}")

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        
        # Calculate gradient reversal coefficient
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        
        # Simple L2 normalization instead of BatchNorm
        # This is much more stable for adversarial training
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        # Apply gradient reversal layer
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        
        # Forward through adversarial layers
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Output layer
        ad_out = self.sigmoid(self.ad_layer3(x))
        
        if self.dual_output:
            fc_out = self.sigmoid(self.fc_layer(x))
            return ad_out, fc_out
        else:
            return ad_out

    def output_num(self):
        return 1
    
    def get_parameters(self):
        # Increased learning rate multiplier for stronger adversarial learning
        return [{"params": self.parameters(), "lr_mult": 50, 'decay_mult': 2}]

# --- Utility Classes ---
class EmbeddingLoader:
    """Utility class to load pre-computed 768-dim embeddings"""
    
    @staticmethod
    def load_embeddings_from_file(file_path):
        """
        Load embeddings from numpy file or torch file
        Expected format: (num_samples, 768)
        """
        if file_path.endswith('.npy'):
            import numpy as np
            embeddings = np.load(file_path)
            return torch.from_numpy(embeddings).float()
        elif file_path.endswith('.pt') or file_path.endswith('.pth'):
            return torch.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    @staticmethod
    def normalize_embeddings(embeddings):
        """L2 normalize embeddings"""
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

# --- EmbeddingFc Model for Pre-computed Embeddings ---
class EmbeddingFc(nn.Module):
    """Model for working directly with pre-computed 768-dim embeddings"""
    
    def __init__(self, class_num=10, use_bottleneck=False, bottleneck_dim=256, new_cls=True):
        super(EmbeddingFc, self).__init__()
        self.class_num = class_num
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        
        # Work directly with 768-dim embeddings
        feature_dim = 768
        
        # Classification layers
        if use_bottleneck and new_cls:
            self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.gvbg = nn.Linear(bottleneck_dim, class_num)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(feature_dim, class_num)
            self.gvbg = nn.Linear(feature_dim, class_num)
            self.__in_features = feature_dim
            
        # Focal layers for GVB
        if new_cls:
            self.focal1 = nn.Linear(class_num, class_num)
            self.focal2 = nn.Linear(class_num, 1)
            
        # Initialize weights
        for module in [self.fc, self.gvbg, getattr(self, 'bottleneck', None), 
                      getattr(self, 'focal1', None), getattr(self, 'focal2', None)]:
            if module is not None:
                module.apply(init_weights)
        
        print(f"✅ EmbeddingFc model initialized for 768-dim embeddings")
        print(f"   - Use bottleneck: {use_bottleneck}")
        print(f"   - Classes: {class_num}")

    def forward(self, embeddings, gvbg=True):
        """
        Forward pass with pre-computed embeddings
        Args:
            embeddings: Pre-computed 768-dim embeddings [batch_size, 768]
        """
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Apply bottleneck if configured
        x = embeddings
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        
        # Classification
        outputs = self.fc(x)
        
        # Focal computation for GVB
        if self.new_cls:
            focal = self.focal2(torch.relu(self.focal1(outputs)))
        else:
            focal = torch.zeros(outputs.size(0), 1, device=outputs.device)
        
        return embeddings, outputs, focal
    
    def get_parameters(self):
        params = []
        
        # All layers with higher learning rate (no pre-trained components)
        if self.use_bottleneck and self.new_cls:
            params.append({"params": self.bottleneck.parameters(), "lr_mult": 10, "decay_mult": 2})
        params.append({"params": self.fc.parameters(), "lr_mult": 10, "decay_mult": 2})
        params.append({"params": self.gvbg.parameters(), "lr_mult": 10, "decay_mult": 2})
        
        if self.new_cls:
            params.append({"params": self.focal1.parameters(), "lr_mult": 10, "decay_mult": 2})
            params.append({"params": self.focal2.parameters(), "lr_mult": 10, "decay_mult": 2})
            
        return params
    
    def output_num(self):
        return self.__in_featurescalc_coeff