"""CPU software check using generated tensors, not an MRI experiment."""
import json

import torch

from med_jepa import CancerJEPA
from med_jepa.roi_v4p3p1.losses import masked_fractional_core_objective


def main():
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    model = CancerJEPA()  # Paper architecture: 128 channels, four residual blocks.
    source = torch.randn(1, 1024, 7, 7, 7)
    history_delta = 0.1 * torch.randn_like(source)
    anchor = 0.95 * source
    target = source + 0.02 * torch.randn_like(source)
    core = torch.zeros(1, 1, 7, 7, 7)
    core[:, :, 2:5, 2:5, 2:5] = 1.0
    ring = torch.zeros_like(core)
    mask = torch.ones_like(core, dtype=torch.bool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    def forward():
        return model.step(
            source, anchor, torch.tensor([21.0]), torch.tensor([150.0]),
            torch.tensor([[1.0, 1.0]]), core, ring,
            history_delta=history_delta, context_mask=mask,
        )

    output = forward()
    objective = masked_fractional_core_objective(output, target, mask, core)
    if not torch.isfinite(objective['total']):
        raise RuntimeError('Non-finite smoke-test loss')
    objective['total'].backward()
    if model.head.weight.grad is None or not model.head.weight.grad.abs().any():
        raise RuntimeError('No learning signal reached the correction head')
    optimizer.step()
    with torch.no_grad():
        updated = forward()
    outside = core.expand_as(source) == 0
    if not torch.equal(updated.predicted[outside], anchor[outside]):
        raise RuntimeError('Correction changed predictions outside the lesion gate')
    print(json.dumps({
        'status': 'passed',
        'input': 'generated tensors; software check only',
        'prediction_shape': list(updated.predicted.shape),
        'backward_pass': 'passed',
        'outside_core_equals_anchor': True,
    }, indent=2))


if __name__ == '__main__':
    main()
