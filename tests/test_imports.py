"""Smoke tests to ensure modules import."""
def test_import_all():
    import importlib
    modules = [
        "data_loader",
        "gnn_backbone",
        "protein_encoder",
        "ban_module",
        "domain_adapt",
        "train_loop",
    ]
    for m in modules:
        importlib.import_module(f"me_drugban.{m}")