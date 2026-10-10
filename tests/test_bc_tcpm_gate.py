import torch

from models.bc_tcpm_gate import BoundaryConsistentTCPMGate


def test_consistent_mode_matches_region_body():
    mask = torch.zeros(1, 1, 17, 17)
    mask[:, :, 3:14, 3:14] = 1
    gate = BoundaryConsistentTCPMGate(kernel_size=3)
    result = gate(mask, mode="consistent")
    assert result.shape == mask.shape
    assert result[:, :, 4:13, 4:13].eq(1).all()
    assert result[:, :, :3].eq(0).all()


def test_legacy_mode_is_identity_and_soft_is_bounded():
    mask = torch.rand(2, 1, 8, 9)
    gate = BoundaryConsistentTCPMGate(kernel_size=3)
    assert torch.equal(gate(mask, mode="legacy"), mask)
    soft = gate(mask, mode="soft", feather=0.25)
    assert soft.shape == mask.shape
    assert float(soft.min()) >= 0.0
    assert float(soft.max()) <= 1.0
