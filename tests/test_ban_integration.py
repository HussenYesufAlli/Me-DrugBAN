def test_ban_predict_smoke():
    from me_drugban.protein_encoder.encoder import ProteinEncoder
    from me_drugban.ban_module.ban import BAN
    pe = ProteinEncoder(output_dim=16)
    ban = BAN(protein_encoder=pe, gnn_backbone=None)
    out = ban.predict("ACDE", [0.1] * 10)
    assert "prot_vec_len" in out and "mol_feats_len" in out