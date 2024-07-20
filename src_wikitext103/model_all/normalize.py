import config
from model_all.plain_layer_norm import plain_layer_norm
from model_all.plain_rms_norm import plain_rms_norm

if config.normalization_type == "Layer Normalization":
    normalize = plain_layer_norm
elif config.normalization_type == "RMS Normalization":
    normalize = plain_rms_norm
else:
    raise Exception()
