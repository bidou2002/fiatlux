"""Private random-number generators for reproducible simulation components."""

from __future__ import annotations

import torch


class RandomGeneratorMixin:
    """Own a device-compatible :class:`torch.Generator`.

    Generator state is part of the Python object, so `copy.deepcopy`,
    `pickle`, and `torch.save` preserve the point reached in the random
    sequence. The explicit state methods also support lightweight checkpoints.
    """

    generator: torch.Generator

    def _configure_generator(
        self,
        device: torch.device | str,
        *,
        seed: int | None,
        generator: torch.Generator | None,
    ) -> None:
        device = torch.device(device)
        if seed is not None and generator is not None:
            raise ValueError("Pass either seed or generator, not both.")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool)
        ):
            raise TypeError("seed must be an integer or None.")
        if generator is not None:
            if not isinstance(generator, torch.Generator):
                raise TypeError("generator must be a torch.Generator or None.")
            if torch.device(generator.device) != device:
                raise ValueError(
                    f"generator device {generator.device} does not match "
                    f"simulation device {device}."
                )
            self.generator = generator
            return

        self.generator = torch.Generator(device=device)
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

    def get_rng_state(self) -> torch.Tensor:
        """Return an independent CPU snapshot of the generator state."""
        return self.generator.get_state().clone()

    def set_rng_state(self, state: torch.Tensor) -> None:
        """Resume the generator from a state returned by :meth:`get_rng_state`."""
        self.generator.set_state(torch.as_tensor(state, device="cpu").clone())
