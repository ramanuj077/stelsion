import ramanuj_model.config as config
from ramanuj_model.architectures.baseline import build_baseline_model
from ramanuj_model.architectures.inceptiontime import build_inceptiontime_model
from ramanuj_model.architectures.minor_axis_attention import build_minor_axis_attention_model
from ramanuj_model.architectures.hybrid import build_hybrid_model

def get_model(architecture_name: str = None, **kwargs):
    """
    Model factory to dynamically retrieve architectures using config toggles.
    Allows easy hyperparameter tuning over architectural specifications.
    """
    if architecture_name is None:
        architecture_name = config.ARCHITECTURE
        
    input_shape = kwargs.get("input_shape", config.INPUT_SHAPE)
    l2_reg = kwargs.get("l2_reg", config.WEIGHT_DECAY)
    dropout_rate = kwargs.get("dropout_rate", config.DROPOUT_RATE)
    
    if architecture_name == "baseline":
        return build_baseline_model(
            input_shape=input_shape,
            l2_reg=l2_reg,
            dropout_rate_fc1=dropout_rate,
            dropout_rate_fc2=dropout_rate
        )
    elif architecture_name == "inceptiontime":
        return build_inceptiontime_model(
            input_shape=input_shape,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate
        )
    elif architecture_name == "minor_axis_attention":
        return build_minor_axis_attention_model(
            input_shape=input_shape,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate
        )
    elif architecture_name == "hybrid":
        return build_hybrid_model(
            input_shape=input_shape,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown architecture selection: {architecture_name}")
