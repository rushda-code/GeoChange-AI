# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:10:35 2026

@author: Admin
"""


"""
GeoChange-AI
Siamese U-Net using MobileNetV2.

TensorFlow / Keras 3

Input:
    Image A: (256, 256, 3)
    Image B: (256, 256, 3)

Output:
    Change probability map: (256, 256, 1)

No training is performed in this file.
"""

import re
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. CONFIGURATION
# ============================================================

IMG_SIZE = 256

DEFAULT_DECODER_FILTERS = [256, 128, 64, 32]


# ============================================================
# 2. ABSOLUTE DIFFERENCE LAYER
# ============================================================

@keras.utils.register_keras_serializable(package="GeoChange")
class AbsDiff(layers.Layer):

    """
    Calculates absolute difference between two feature maps.
    """

    def call(self, inputs):

        feature_a, feature_b = inputs

        return keras.ops.abs(feature_a - feature_b)

    def get_config(self):

        return super().get_config()


# ============================================================
# 3. BUILD MOBILENETV2 ENCODER
# ============================================================

def build_encoder(
    verbose=True,
    encoder_weights="imagenet"
):

    """
    Build MobileNetV2 and select five feature maps.

    encoder_weights:
        "imagenet" = initialize using ImageNet weights.
        None       = initialize without pretrained weights.

    For deployment, use None and load the exported
    trained model weights afterwards.
    """

    print("\n" + "=" * 55)
    print("BUILDING MOBILENETV2 ENCODER")
    print("=" * 55)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=encoder_weights
    )

    if encoder_weights == "imagenet":

        print("ImageNet weights requested.")

    elif encoder_weights is None:

        print("Encoder initialized without ImageNet weights.")
        print("Use load_weights() to load the trained weights.")

    else:

        print("Encoder weights:", encoder_weights)

    # --------------------------------------------------------
    # FIND EXPANSION-RELU CANDIDATES
    # --------------------------------------------------------

    target_scales = [128, 64, 32, 16]

    candidates = {
        scale: []
        for scale in target_scales
    }

    for layer in base.layers:

        if not layer.name.endswith("_expand_relu"):
            continue

        shape = tuple(layer.output.shape)

        height = shape[1]
        width = shape[2]

        if height != width:
            continue

        if height not in candidates:
            continue

        match = re.search(
            r"block_(\d+)_expand_relu",
            layer.name
        )

        if match is None:
            continue

        block_number = int(match.group(1))

        candidates[height].append(
            (block_number, layer)
        )

    # --------------------------------------------------------
    # SELECT ONE LAYER PER SCALE
    # --------------------------------------------------------

    selected_layers = []
    selected_information = []

    print("\nSELECTED SKIP CONNECTIONS")
    print("-" * 80)

    for scale in target_scales:

        if not candidates[scale]:

            raise ValueError(
                f"No expansion-ReLU candidate found "
                f"at {scale}x{scale}."
            )

        block_number, chosen_layer = max(
            candidates[scale],
            key=lambda item: item[0]
        )

        shape = tuple(chosen_layer.output.shape)

        selected_layers.append(
            chosen_layer.output
        )

        selected_information.append({
            "scale": scale,
            "layer": chosen_layer.name,
            "shape": shape,
            "channels": shape[-1],
            "block": block_number
        })

        print(
            f"{scale}x{scale} | "
            f"{chosen_layer.name} | "
            f"{shape}"
        )

    # --------------------------------------------------------
    # FINAL 8x8 FEATURE MAP
    # --------------------------------------------------------

    final_feature = base.output

    final_shape = tuple(final_feature.shape)

    if final_shape[1:3] != (8, 8):

        raise ValueError(
            f"Expected 8x8 final feature map, "
            f"got {final_shape}."
        )

    selected_layers.append(final_feature)

    selected_information.append({
        "scale": 8,
        "layer": base.layers[-1].name,
        "shape": final_shape,
        "channels": final_shape[-1],
        "block": "Final"
    })

    print(
        f"8x8 | "
        f"{base.layers[-1].name} | "
        f"{final_shape}"
    )

    # --------------------------------------------------------
    # FEATURE EXTRACTOR MODEL
    # --------------------------------------------------------

    encoder = keras.Model(
        inputs=base.input,
        outputs=selected_layers,
        name="mobilenetv2_encoder"
    )

    print("\nMobileNetV2 encoder created successfully.")

    return encoder, selected_information


# ============================================================
# 4. DECODER CONVOLUTION BLOCK
# ============================================================

def conv_block(x, filters, name):

    """
    Two convolution blocks.

    Each block:
        Conv2D -> BatchNormalization -> ReLU
    """

    for index in range(2):

        block_name = f"{name}_conv{index + 1}"

        x = layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name=f"{block_name}_conv"
        )(x)

        x = layers.BatchNormalization(
            name=f"{block_name}_bn"
        )(x)

        x = layers.ReLU(
            name=f"{block_name}_relu"
        )(x)

    return x


# ============================================================
# 5. SHARED DECODER
# ============================================================

def build_decoder(features, decoder_filters):

    """
    Common decoder for Siamese U-Net and single-image U-Net.

    Features arrive in this order:
        128, 64, 32, 16, 8

    Decoder moves from 8x8 to 256x256.
    """

    if len(decoder_filters) != 4:

        raise ValueError(
            "decoder_filters must contain exactly four values."
        )

    f128, f64, f32, f16, f8 = features

    x = f8

    # --------------------------------------------------------
    # 8x8 -> 16x16
    # --------------------------------------------------------

    x = layers.UpSampling2D(
        size=(2, 2),
        name="upsample_16"
    )(x)

    x = layers.Concatenate(
        name="concat_16"
    )([x, f16])

    x = conv_block(
        x,
        decoder_filters[0],
        "decoder_16"
    )

    # --------------------------------------------------------
    # 16x16 -> 32x32
    # --------------------------------------------------------

    x = layers.UpSampling2D(
        size=(2, 2),
        name="upsample_32"
    )(x)

    x = layers.Concatenate(
        name="concat_32"
    )([x, f32])

    x = conv_block(
        x,
        decoder_filters[1],
        "decoder_32"
    )

    # --------------------------------------------------------
    # 32x32 -> 64x64
    # --------------------------------------------------------

    x = layers.UpSampling2D(
        size=(2, 2),
        name="upsample_64"
    )(x)

    x = layers.Concatenate(
        name="concat_64"
    )([x, f64])

    x = conv_block(
        x,
        decoder_filters[2],
        "decoder_64"
    )

    # --------------------------------------------------------
    # 64x64 -> 128x128
    # --------------------------------------------------------

    x = layers.UpSampling2D(
        size=(2, 2),
        name="upsample_128"
    )(x)

    x = layers.Concatenate(
        name="concat_128"
    )([x, f128])

    x = conv_block(
        x,
        decoder_filters[3],
        "decoder_128"
    )

    # --------------------------------------------------------
    # 128x128 -> 256x256
    # --------------------------------------------------------

    x = layers.UpSampling2D(
        size=(2, 2),
        name="upsample_256"
    )(x)

    x = conv_block(
        x,
        decoder_filters[3],
        "decoder_256"
    )

    # --------------------------------------------------------
    # FINAL CHANGE PROBABILITY
    # --------------------------------------------------------

    output = layers.Conv2D(
        filters=1,
        kernel_size=1,
        activation="sigmoid",
        dtype="float32",
        name="change_probability"
    )(x)

    return output


# ============================================================
# 6. SIAMESE U-NET
# ============================================================

def build_siamese_unet(
    freeze_encoder=False,
    decoder_filters=DEFAULT_DECODER_FILTERS,
    use_mixed_precision=False,
    verbose=True,
    return_components=False,
    encoder_weights="imagenet"
):

    """
    Build the complete Siamese U-Net.

    Two inputs:
        Image A
        Image B

    One shared MobileNetV2 encoder.

    Absolute difference between corresponding features.

    One shared decoder.

    One sigmoid output.

    encoder_weights:
        "imagenet" = normal construction.
        None       = app deployment construction.
    """

    print("\n" + "=" * 55)
    print("BUILDING SIAMESE U-NET")
    print("=" * 55)

    # --------------------------------------------------------
    # MIXED PRECISION
    # --------------------------------------------------------

    if use_mixed_precision:

        keras.mixed_precision.set_global_policy(
            "mixed_float16"
        )

        print("Mixed precision: ENABLED")

    else:

        keras.mixed_precision.set_global_policy(
            "float32"
        )

        print("Mixed precision: DISABLED")

    # --------------------------------------------------------
    # BUILD ONE SHARED ENCODER
    # --------------------------------------------------------

    encoder, skip_information = build_encoder(
        verbose=verbose,
        encoder_weights=encoder_weights
    )

    encoder.trainable = not freeze_encoder

    print("Encoder trainable:", encoder.trainable)

    # --------------------------------------------------------
    # TWO INPUTS
    # --------------------------------------------------------

    input_a = keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="imageA"
    )

    input_b = keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="imageB"
    )

    # --------------------------------------------------------
    # SAME ENCODER OBJECT FOR BOTH INPUTS
    # --------------------------------------------------------

    if freeze_encoder:

        features_a = encoder(
            input_a,
            training=False
        )

        features_b = encoder(
            input_b,
            training=False
        )

    else:

        features_a = encoder(input_a)

        features_b = encoder(input_b)

    # --------------------------------------------------------
    # ABSOLUTE DIFFERENCE AT ALL FIVE SCALES
    # --------------------------------------------------------

    difference_features = []

    for scale, feature_a, feature_b in zip(
        [128, 64, 32, 16, 8],
        features_a,
        features_b
    ):

        difference = AbsDiff(
            name=f"abs_diff_{scale}"
        )([feature_a, feature_b])

        difference_features.append(difference)

    # --------------------------------------------------------
    # DECODER
    # --------------------------------------------------------

    output = build_decoder(
        difference_features,
        decoder_filters
    )

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    model = keras.Model(
        inputs=[input_a, input_b],
        outputs=output,
        name="GeoChange_Siamese_UNet"
    )

    print("\nSiamese U-Net created successfully.")
    print("Input shapes:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Parameters:", model.count_params())

    if return_components:

        diff_model = keras.Model(
            inputs=[input_a, input_b],
            outputs=difference_features,
            name="feature_difference_checker"
        )

        return (
            model,
            encoder,
            diff_model,
            skip_information
        )

    return model


# ============================================================
# 7. SINGLE-IMAGE U-NET BACKUP
# ============================================================

def build_unet(
    freeze_encoder=False,
    decoder_filters=DEFAULT_DECODER_FILTERS,
    use_mixed_precision=False,
    encoder_weights="imagenet"
):

    """
    Single-image U-Net backup architecture.

    Uses the same MobileNetV2 encoder and decoder.
    """

    if use_mixed_precision:

        keras.mixed_precision.set_global_policy(
            "mixed_float16"
        )

    else:

        keras.mixed_precision.set_global_policy(
            "float32"
        )

    encoder, _ = build_encoder(
        verbose=False,
        encoder_weights=encoder_weights
    )

    encoder.trainable = not freeze_encoder

    image = keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="image"
    )

    if freeze_encoder:

        features = encoder(
            image,
            training=False
        )

    else:

        features = encoder(image)

    output = build_decoder(
        features,
        decoder_filters
    )

    model = keras.Model(
        inputs=image,
        outputs=output,
        name="GeoChange_Single_UNet"
    )

    print("\nSingle-image U-Net created.")
    print("Output shape:", model.output_shape)

    return model


# ============================================================
# 8. QUICK ARCHITECTURE CHECK
# ============================================================

if __name__ == "__main__":

    print("\nTesting deployment architecture...")

    model = build_siamese_unet(
        encoder_weights=None,
        verbose=False
    )

    print("\nARCHITECTURE CHECK")
    print("-" * 55)

    print("Input:", model.input_shape)
    print("Output:", model.output_shape)
    print("Parameters:", model.count_params())

    assert model.input_shape == [
        (None, 256, 256, 3),
        (None, 256, 256, 3)
    ]

    assert model.output_shape == (
        None, 256, 256, 1
    )

    assert model.count_params() == 8058977

    print("\nPASS: Deployment architecture checks passed.")
