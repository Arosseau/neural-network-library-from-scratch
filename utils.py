import activation_functions
import initializers
import loss_functions


def get_initializer(name):
    return getattr(
        initializers,
        initializers.initializer_fs[name.lower()]
    )


def get_activation(name, d=""):
    return getattr(
        activation_functions,
        d + activation_functions.activation_fs[name.lower()]
    )


def get_loss(name, d=""):
    return getattr(
        loss_functions,
        d + loss_functions.loss_fs[name.lower()]
    )