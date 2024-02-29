import keras
import keras.ops as ops

logger = logging.getLogger(__name__)

def get_callbacks(config: Dict[str, str], path_extend=False) -> List[Callback]:
    callbacks = []

    if "callbacks" in config:
        if path_extend:
            save_data = os.path.join(config["save_loc"], path_extend)
        else:
            save_data = config["save_loc"]
        config = config["callbacks"]
    else:
        return []

    if "ModelCheckpoint" in config:
        config["ModelCheckpoint"]["filepath"] = os.path.join(
            save_data, config["ModelCheckpoint"]["filepath"]
        )
        callbacks.append(ModelCheckpoint(**config["ModelCheckpoint"]))
        logger.info("... loaded Checkpointer")

    if "EarlyStopping" in config:
        callbacks.append(EarlyStopping(**config["EarlyStopping"]))
        logger.info("... loaded EarlyStopping")

    # LearningRateTracker(),  ## ReduceLROnPlateau does this already, use when supplying custom LR annealer

    if "ReduceLROnPlateau" in config:
        callbacks.append(ReduceLROnPlateau(**config["ReduceLROnPlateau"]))
        logger.info("... loaded ReduceLROnPlateau")

    if "CSVLogger" in config:
        config["CSVLogger"]["filename"] = os.path.join(
            save_data, config["CSVLogger"]["filename"]
        )
        callbacks.append(CSVLogger(**config["CSVLogger"]))
        logger.info("... loaded CSVLogger")

    if "LearningRateScheduler" in config:
        drop = config["LearningRateScheduler"]["drop"]
        epochs_drop = config["LearningRateScheduler"]["epochs_drop"]
        f = partial(step_decay, drop=drop, epochs_drop=epochs_drop)
        callbacks.append(LearningRateScheduler(f))
        callbacks.append(LearningRateTracker())

    return callbacks


def step_decay(epoch, drop=0.2, epochs_drop=5.0, init_lr=0.001):
    lrate = init_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None) -> None:
        logs = logs or {}
        logs["lr"] = K.get_value(self.model.optimizer.lr)


class ReportEpoch(keras.callbacks.Callback):
    def __init__(self, epoch_var):
        self.epoch_var = epoch_var

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_var.assign_add(1)