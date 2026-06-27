"""Model factory: a U-Net (segmentation_models_pytorch) for N-band input."""

import segmentation_models_pytorch as smp

from . import NUM_CLASSES


def build_model(classes=NUM_CLASSES, in_channels=4,
                encoder="resnet34", encoder_weights="imagenet"):
    """U-Net with an ImageNet-pretrained encoder.

    smp adapts the first conv layer automatically when in_channels != 3,
    so the NIR band (B08) is supported out of the box. Pass
    encoder_weights=None for offline use (e.g. the smoke test).
    """
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
