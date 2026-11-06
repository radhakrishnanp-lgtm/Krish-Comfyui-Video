"""Linear <-> display transfer functions for EXR data (torch only, no ComfyUI imports).

Models like LTX expect display-referred 0..1 images; EXRs are normally scene-linear and may contain values above 1
or below 0. These helpers convert for the model and back, extending the curves symmetrically so negatives survive.
Only the transfer function changes: primaries are never converted, and the EXR header is never touched.
"""
import torch

TRANSFORMS = ["sRGB", "Rec.709", "gamma 2.2", "gamma 2.4", "none", "ACEScct (log, invertible)"]
HIGHLIGHTS = ["clip", "soft_knee"]


def _srgb_oetf(x):
    a = x.abs()
    return torch.sign(x) * torch.where(a <= 0.0031308, 12.92 * a, 1.055 * a.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def _srgb_eotf(y):
    a = y.abs()
    return torch.sign(y) * torch.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def _rec709_oetf(x):
    a = x.abs()
    return torch.sign(x) * torch.where(a < 0.018, 4.5 * a, 1.099 * a.clamp_min(1e-12) ** 0.45 - 0.099)


def _rec709_eotf(y):
    a = y.abs()
    return torch.sign(y) * torch.where(a < 0.081, a / 4.5, ((a + 0.099) / 1.099) ** (1 / 0.45))


def _gamma_oetf(g):
    return lambda x: torch.sign(x) * x.abs() ** (1 / g)


def _gamma_eotf(g):
    return lambda y: torch.sign(y) * y.abs() ** g


# ACEScct (Academy S-2016-001): a log curve whose 0..1 covers scene-linear -0.0069 .. 222.9, so highlights survive
# the trip into a model and can be recovered exactly. Curve only: no primaries conversion.
_CCT_A, _CCT_B = 10.5402377416545, 0.0729055341958355
_CCT_LIN_CUT, _CCT_ENC_CUT = 0.0078125, 0.155251141552511
ACESCCT_MAX_LINEAR = 2.0 ** (17.52 - 9.72)


def acescct_encode(x):
    return torch.where(x <= _CCT_LIN_CUT, _CCT_A * x + _CCT_B,
                       (torch.log2(x.clamp_min(_CCT_LIN_CUT)) + 9.72) / 17.52)


def acescct_decode(y):
    return torch.where(y <= _CCT_ENC_CUT, (y - _CCT_B) / _CCT_A, 2.0 ** (y.clamp_min(_CCT_ENC_CUT) * 17.52 - 9.72))


def _curves(name):
    if name.startswith("ACEScct"):
        return acescct_encode, acescct_decode
    if name == "sRGB":
        return _srgb_oetf, _srgb_eotf
    if name == "Rec.709":
        return _rec709_oetf, _rec709_eotf
    if name == "gamma 2.2":
        return _gamma_oetf(2.2), _gamma_eotf(2.2)
    if name == "gamma 2.4":
        return _gamma_oetf(2.4), _gamma_eotf(2.4)
    if name == "none":
        return (lambda x: x), (lambda x: x)
    raise ValueError(f"unknown transform: {name}")


def soft_knee(x, knee=0.8):
    """Roll scene values above `knee` off toward 1.0 (monotonic, continuous, slope 1 at the knee).

    Hyperbolic, so very bright values stay distinguishable instead of all landing on exactly 1.0.
    """
    k = float(knee)
    c = 1 - k
    over = (x - k).clamp_min(0)
    return torch.where(x <= k, x, k + c * over / (over + c))


def linear_to_display(images, transform="sRGB", exposure_stops=0.0, highlights="clip"):
    """Scene-linear -> display-referred for the model. 'clip' ends with a hard 0..1 clamp."""
    x = images.float() * (2.0 ** float(exposure_stops))
    if highlights == "soft_knee" and not transform.startswith("ACEScct"):   # the log already holds the range
        x = soft_knee(x.clamp_min(0))
    y = _curves(transform)[0](x)
    return y.clamp(0, 1) if highlights == "clip" else y.clamp_min(0)


def soft_knee_inverse(z, knee=0.8):
    """Exact inverse of soft_knee. Values at or above 1.0 decode to a large finite maximum instead of infinity."""
    k = float(knee)
    c = 1 - k
    over = (z - k).clamp(0, c * (1 - 1e-5))
    return torch.where(z <= k, z, k + c * over / (c - over))


def display_to_linear(images, transform="sRGB", exposure_stops=0.0, highlights="clip"):
    """Display-referred -> scene-linear. No clamping. highlights='soft_knee' also undoes the soft knee that
    linear_to_display applied, so linear_to_display(..., 'soft_knee') and this are inverses of each other."""
    lin = _curves(transform)[1](images.float())
    if highlights == "soft_knee" and not transform.startswith("ACEScct"):
        lin = soft_knee_inverse(lin)
    return lin * (2.0 ** -float(exposure_stops))


def merge_outside_edit(original_linear, original_encoded, edited_encoded, transform="sRGB", exposure_stops=0.0,
                       highlights="clip", tolerance=1e-6):
    """Decode only what a model changed; everything it left alone comes back from the ORIGINAL linear pixels.

    original_encoded is the exact image that was given to the model workflow. Pixels where the edited result equals it
    (within `tolerance`) are restored bit-exactly from original_linear instead of passing through the encode/decode curve.
    Returns (linear image, changed mask [N,H,W]).
    """
    dev = edited_encoded.device
    ol, oe = original_linear.to(dev).float(), original_encoded.to(dev).float()
    ee = edited_encoded.float()[..., :3]
    if not (ol.shape[:3] == oe.shape[:3] == ee.shape[:3]):
        raise ValueError(f"size mismatch: original {tuple(ol.shape[1:3])}, encoded {tuple(oe.shape[1:3])}, "
                         f"edited {tuple(ee.shape[1:3])}; all three must be the same size and frame count")
    changed = ((ee - oe[..., :3]).abs() > float(tolerance)).any(dim=-1, keepdim=True)
    decoded = display_to_linear(ee, transform, exposure_stops, highlights)
    out = torch.where(changed, decoded, ol[..., :3])
    return out, changed.squeeze(-1).float()
