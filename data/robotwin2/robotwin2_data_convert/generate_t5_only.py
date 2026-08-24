#!/usr/bin/env python3
"""
Standalone script to generate T5 embeddings for existing RobotWin dataset.
This script only generates T5 embeddings, assuming videos/qpos/metas already exist.
"""

import sys
import os
from pathlib import Path
import yaml
import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# This must be done before importing any modules that use CUDA
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Start method already set
    if multiprocessing.get_start_method() != 'spawn':
        print(f"Warning: Multiprocessing start method is '{multiprocessing.get_start_method()}', "
              f"but 'spawn' is required for CUDA. This may cause issues.")

# Add parent directories to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from robotwin_converter import RobotWinConverter

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    script_dir = Path(__file__).parent
    config_file = script_dir / "config.yml"
    
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(str(config_file))
    
    # Validate required paths
    target_root = config.get('target_root', '')
    wan_repo_path = config.get('wan_repo_path', '')
    
    if not target_root:
        print("Error: target_root is not set in config.yml")
        sys.exit(1)
    
    if not wan_repo_path:
        print("Error: wan_repo_path is not set in config.yml")
        sys.exit(1)
    
    target_path = Path(target_root)
    wan_path = Path(wan_repo_path)
    
    if not target_path.exists():
        print(f"Error: Target root not found: {target_root}")
        sys.exit(1)
    
    if not wan_path.exists():
        print(f"Error: WAN repo path not found: {wan_repo_path}")
        sys.exit(1)
    
    # Check T5 model files
    t5_model_path = wan_path / "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer_path = wan_path / "google/umt5-xxl"
    
    if not t5_model_path.exists():
        print(f"Error: T5 model not found: {t5_model_path}")
        sys.exit(1)
    
    if not t5_tokenizer_path.exists():
        print(f"Error: T5 tokenizer not found: {t5_tokenizer_path}")
        sys.exit(1)
    
    # Check available GPUs
    try:
        import torch
        if not torch.cuda.is_available():
            print("Error: CUDA is not available. T5 embeddings generation requires GPU.")
            sys.exit(1)
        
        num_available_gpus = torch.cuda.device_count()
        configured_devices = config.get('cuda_devices', ['0'])
        
        # Filter to only use available GPUs
        available_devices = [str(i) for i in range(num_available_gpus)]
        valid_devices = [d for d in configured_devices if d in available_devices]
        
        if not valid_devices:
            print(f"Warning: None of the configured devices {configured_devices} are available.")
            print(f"Available GPUs: {available_devices}")
            print(f"Using all available GPUs: {available_devices}")
            valid_devices = available_devices
        
        # Update config with valid devices
        config['cuda_devices'] = valid_devices
        
        print("=" * 60)
        print("T5 Embeddings Generation")
        print("=" * 60)
        print(f"Target Root: {target_root}")
        print(f"WAN Repo Path: {wan_repo_path}")
        print(f"T5 Max Length: {config.get('t5_max_length', 512)}")
        print(f"Available GPUs: {num_available_gpus}")
        print(f"Configured Devices: {configured_devices}")
        print(f"Using Devices: {valid_devices}")
        print("=" * 60)
        print()
        
    except ImportError:
        print("Warning: PyTorch not available, cannot check GPU count")
        print("Will use configured devices:", config.get('cuda_devices', ['0']))
        print()
    
    # Update config to enable T5 embeddings if not already enabled
    original_enable_t5 = config.get('enable_t5_embeddings', False)
    original_source_root = config.get('source_root', '')
    
    # Ensure source_root exists for validation (use target_root if source_root not set)
    if not original_source_root or not Path(original_source_root).exists():
        print("Note: source_root not set or doesn't exist, using target_root for validation")
        config['source_root'] = str(target_root)
    
    if not original_enable_t5:
        print("Note: enable_t5_embeddings is False in config.yml")
        print("Temporarily enabling it for T5 embeddings generation...")
        config['enable_t5_embeddings'] = True
    
    # Save updated config back to file temporarily
    config_modified = (not original_enable_t5) or (not original_source_root or not Path(original_source_root).exists())
    if config_modified:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print("Config file temporarily updated.")
    
    # Create converter instance with config file path
    converter = RobotWinConverter(str(config_file))
    
    # Ensure enable_t5_embeddings is True (double check)
    converter.config['enable_t5_embeddings'] = True
    
    # Only generate T5 embeddings
    print("Collecting meta files for T5 processing...")
    converter.process_t5_embeddings_parallel()
    
    print()
    print("=" * 60)
    print("T5 Embeddings Generation Completed!")
    print("=" * 60)
    
    # Count generated files
    t5_count = len(list(target_path.rglob("umt5_wan/*.pt")))
    meta_count = len(list(target_path.rglob("metas/*.txt")))
    
    print(f"Generated T5 embeddings: {t5_count}")
    print(f"Total meta files: {meta_count}")
    if meta_count > 0:
        print(f"Coverage: {t5_count * 100 // meta_count}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
