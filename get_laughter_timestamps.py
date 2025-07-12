# Extract laughter timestamps only (no audio file generation)
# Usage: python get_laughter_timestamps.py --input_audio_file=sample.m4a --threshold=0.2 --min_length=0.1 --padding=0.3 --output_json=laughter_timestamps.json

import os, sys, pickle, time, librosa, argparse, torch, numpy as np, pandas as pd, scipy, json
from tqdm import tqdm
import tgt
sys.path.append('./utils/')
import laugh_segmenter
import models, configs
import dataset_utils, audio_utils, data_loaders, torch_utils
from tqdm import tqdm
from torch import optim, nn
from functools import partial
from distutils.util import strtobool

sample_rate = 8000

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='checkpoints/in_use/resnet_with_augmentation')
parser.add_argument('--config', type=str, default='resnet_with_augmentation')
parser.add_argument('--threshold', type=str, default='0.5')
parser.add_argument('--min_length', type=str, default='0.2')
parser.add_argument('--padding', type=str, default='0.0', help='Add padding (seconds) before and after each laugh segment')
parser.add_argument('--input_audio_file', required=True, type=str)
parser.add_argument('--output_json', type=str, default='laughter_timestamps.json')

args = parser.parse_args()

if __name__ == '__main__':
    model_path = args.model_path
    config = configs.CONFIG_MAP[args.config]
    audio_path = args.input_audio_file
    threshold = float(args.threshold)
    min_length = float(args.min_length)
    padding = float(args.padding)
    output_json = args.output_json

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    ##### Load the Model

    model = config['model'](dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
    feature_fn = config['feature_fn']
    model.set_device(device)

    if os.path.exists(model_path):
        torch_utils.load_checkpoint(model_path+'/best.pth.tar', model)
        model.eval()
    else:
        raise Exception(f"Model checkpoint not found at {model_path}")
        
    ##### Load the audio file and features
        
    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=audio_path, feature_fn=feature_fn, sr=sample_rate)

    collate_fn=partial(audio_utils.pad_sequences_with_labels,
                            expand_channel_dim=config['expand_channel_dim'])

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset, num_workers=4, batch_size=8, shuffle=False, collate_fn=collate_fn)


    ##### Make Predictions

    probs = []
    for model_inputs, _ in tqdm(inference_generator):
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        if len(preds.shape)==0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)

    file_length = audio_utils.get_audio_length(audio_path)

    fps = len(probs)/float(file_length)

    probs = laugh_segmenter.lowpass(probs)
    instances = laugh_segmenter.get_laughter_instances(probs, threshold=threshold, min_length=float(args.min_length), fps=fps)

    print(); print("found %d laughs." % (len(instances)))

    # Create timestamp data with padding applied
    timestamp_data = []
    for index, instance in enumerate(instances):
        # Apply padding to the instance
        padded_start = max(0, instance[0] - padding)
        padded_end = min(file_length, instance[1] + padding)
        
        timestamp_data.append({
            "laugh_id": index,
            "start": padded_start,
            "end": padded_end,
            "duration": padded_end - padded_start,
            "original_start": instance[0],
            "original_end": instance[1],
            "original_duration": instance[1] - instance[0]
        })

    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(timestamp_data, f, indent=2)

    print(f"Saved {len(instances)} laughter timestamps to {output_json}")
    
    # Also print summary
    if len(instances) > 0:
        print("\nLaughter segments found:")
        for item in timestamp_data:
            print(f"Laugh {item['laugh_id']}: {item['start']:.2f}s - {item['end']:.2f}s (duration: {item['duration']:.2f}s)")
