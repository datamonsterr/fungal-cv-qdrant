from src.prepare.dataset import prepare_dataset


def reformat_dataset(create_hierarchical: bool = True):
    del create_hierarchical
    return prepare_dataset(source_collections=["curated"])


if __name__ == "__main__":
    reformat_dataset()
