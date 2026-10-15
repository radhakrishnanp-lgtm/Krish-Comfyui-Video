# Krish Frame Conform (ComfyUI, LTX)

Makes LTX video-to-video return **exactly the same frame count, order and size** as the input.

LTX needs 8n+1 frames and width/height divisible by 32. ComfyUI's LTX nodes crop or round anything else,
so frames get lost and the duration drifts. These nodes pad onto the grid before sampling and trim after.

## Install
Copy or symlink `krish_frame_conform/` into `ComfyUI/custom_nodes/` and restart.

## Workflow
```
Load Video -> Krish LTX Frame Prep -> (LTX conditioning/sampler/VAE decode) -> Krish LTX Frame Restore -> Parity Check -> Save
                  |  conform_info ------------------------------------------^
                  |  num_frames / width / height -> LTX latent size inputs
```
- **Prep** pads the tail (repeat or ping-pong) and the edges (reflect). Original pixels are never resampled.
- **Restore** trims the padding. If the model returned another frame count or size, it retimes/resizes back and
  reports what it fixed. Optional `source_images` + `mask` keep unedited regions pixel-exact from the source.
- **Parity Check** errors if count or size ever differs from the source.
- Set the video combine fps to the source fps. These nodes do not change timing.

Note: the VAE round trip changes pixel values inside the edited area. These nodes guarantee count, order,
size and timing. Use the mask input for pixel-exact preservation elsewhere.

## Tests
`pip install torch pytest && pytest tests`
