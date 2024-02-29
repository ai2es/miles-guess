import sklearn.model_selection as ms


def load_splitter(splitter, n_splits=1, random_state=1000, train_size=0.9):
    choices = [
        "GroupKFold",
        "GroupShuffleSplit",
        "ShuffleSplit",
        "StratifiedKFold",
        "StratifiedShuffleSplit",
        "StratifiedGroupKFold",
        "TimeSeriesSplit",
    ]

    if splitter == "GroupKFold":
        return ms.GroupKFold(n_splits=n_splits)
    elif splitter == "GroupShuffleSplit":
        return ms.GroupShuffleSplit(
            n_splits=n_splits, random_state=random_state, train_size=train_size
        )
    elif splitter == "ShuffleSplit":
        return ms.ShuffleSplit(
            n_splits=n_splits, random_state=random_state, train_size=train_size
        )
    elif splitter == "StratifiedKFold":
        return ms.StratifiedKFold(
            n_splits=n_splits, random_state=random_state
        )
    elif splitter == "StratifiedShuffleSplit":
        return ms.StratifiedShuffleSplit(
            n_splits=n_splits, random_state=random_state, train_size=train_size
        )
    elif splitter == "StratifiedGroupKFold":
        return ms.StratifiedGroupKFold(
            n_splits=n_splits, random_state=random_state,
        )
    elif splitter == "TimeSeriesSplit":
        return ms.TimeSeriesSplit(
            n_splits=n_splits,
        )
    else:
        raise OSError(
            f"I did not recognize your split type {splitter}. Choose from {choices}"
        )
