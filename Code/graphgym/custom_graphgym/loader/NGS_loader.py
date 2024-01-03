from Code.datasetClass import InOrgMatDatasets
from torch_geometric.graphgym.register import register_loader


@register_loader('NGS-dataset')
def load_NGS_dataset(format, name, dataset_dir):
    if format == 'PyG':
        dataset_raw = InOrgMatDatasets(dataset=name, root=dataset_dir)
        return dataset_raw
