import argparse
import cv2
import torch
import numpy as np
from models.crnn import CRNN
from config import config
from utils.vocab import vocab  # Import the vocab instance


def decode_predictions(preds, vocab_obj):
    """Convert model output to readable text"""
    pred_texts = []
    # preds shape: (seq_len, batch_size, num_classes)
    _, max_indices = torch.max(preds, 2)  # Get most likely characters

    for indices in max_indices.transpose(0, 1):  # Iterate through batch
        # CTC decoding: merge repeated chars and remove blank
        decoded = []
        prev_char = None
        for idx in indices:
            idx = idx.item()
            if idx != vocab_obj.blank_char and idx != prev_char:
                if idx < len(vocab_obj.get_vocab()):
                    decoded.append(vocab_obj.idx2char[idx])
            prev_char = idx
        pred_texts.append(''.join(decoded))

    return pred_texts


def test_image(model, image_path, device, vocab_obj):
    """Test a single image"""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0
        img = (img - config.MEAN) / config.STD
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        img = img.to(device)

        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(img)
            pred_text = decode_predictions(outputs, vocab_obj)[0]

        return pred_text
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return ""


def main():
    parser = argparse.ArgumentParser(description='Test handwritten text recognition model')
    parser.add_argument('--image', type=str, required=True, help='Path to image file to test')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pt file)')
    args = parser.parse_args()

    # Initialize device
    device = torch.device(config.DEVICE)

    # Load model
    model = CRNN(
        img_height=config.IMG_HEIGHT,
        num_channels=1,
        num_classes=len(config.VOCAB) + 1,  # +1 for CTC blank
        hidden_size=config.HIDDEN_SIZE,
        num_lstm_layers=config.NUM_LSTM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device))

    # Test image
    pred_text = test_image(model, args.image, device, vocab)  # Use vocab instance
    print(f"Predicted text: {pred_text}")


if __name__ == "__main__":
    main()