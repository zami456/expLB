# ruff: noqa
# fmt: off
"""The reference steering hook, verbatim, for tests/test_steering.py to compare against.

`DeceptionSteeringHook` copied from ARENA's "[1.3.1] Linear Probes (solutions)" notebook, section 4
("Probing for Deception"), the cell that implements it (the user's copy:
`1_3_1_Linear_Probes_solutions (2).ipynb`). The notebook imports torch as `t`, so that import is
added here; nothing else is changed. Lint and formatting are off so the copy stays verbatim.
"""

import torch as t


class DeceptionSteeringHook:
    """Hook that adds a steering vector to a model's residual stream during the forward pass."""

    def __init__(self, steering_vector, layer, steering_coef, apply_to_all_tokens=True):
        self.steering_vector = steering_vector
        self.layer = layer
        self.steering_coef = steering_coef
        self.apply_to_all_tokens = apply_to_all_tokens
        self.hook = None

    def _hook_fn(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        v = self.steering_vector.to(hidden_states.device, dtype=hidden_states.dtype)
        v_normed = v / (v.norm() + 1e-8)

        if self.apply_to_all_tokens:
            norm = t.norm(hidden_states, dim=-1, keepdim=True)
            hidden_states = hidden_states + self.steering_coef * norm * v_normed
        else:
            norm = t.norm(hidden_states[:, -1, :], dim=-1, keepdim=True)
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + self.steering_coef * norm * v_normed

        return (hidden_states,) + output[1:] if isinstance(output, tuple) else hidden_states

    def enable(self, model):
        self.hook = model.model.layers[self.layer].register_forward_hook(self._hook_fn)

    def disable(self):
        if self.hook:
            self.hook.remove()
            self.hook = None
