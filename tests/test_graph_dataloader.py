from me_drugban.data_loader.dataloader import MoleculeDataset

def test_graph_loader():
    dset = MoleculeDataset("data/bindingdb_sample/train.csv")
    sample = dset[0]
    g = sample["graph"]
    print("Graph:", g)
    assert g is not None
    assert g.number_of_nodes() > 0
    assert g.number_of_edges() > 0
    assert "h" in g.ndata  # Atom features

def test_protein_encoding():
    dset = MoleculeDataset("data/bindingdb_sample/train.csv")
    sample = dset[0]
    enc = sample["protein_encoded"]
    print("Protein encoded:", enc[:20], "len=", len(enc))
    assert isinstance(enc, list)
    assert len(enc) == 1200