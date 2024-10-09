from pathlib import Path

import mlx.core as mx

from mflux.config.runtime_config import RuntimeConfig
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.image_util import ImageUtil


class StepwiseHandler:
    def __init__(
        self,
        flux,
        config: RuntimeConfig,
        seed: int,
        prompt: str,
        time_steps,
        output_dir: Path | None = None,
    ):
        self.flux = flux
        self.config = config
        self.seed = seed
        self.prompt = prompt
        self.output_dir = output_dir
        self.time_steps = time_steps
        self.step_wise_images = []

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_step(self, step: int, latents: mx.array):
        if self.output_dir:
            unpack_latents = ArrayUtil.unpack_latents(latents, self.config.height, self.config.width)
            stepwise_decoded = self.flux.vae.decode(unpack_latents)
            stepwise_img = ImageUtil.to_image(
                decoded_latents=stepwise_decoded,
                seed=self.seed,
                prompt=self.prompt,
                quantization=self.flux.bits,
                generation_time=self.time_steps.format_dict["elapsed"],
                lora_paths=self.flux.lora_paths,
                lora_scales=self.flux.lora_scales,
                config=self.config,
            )
            self.step_wise_images.append(stepwise_img)

            stepwise_img.save(
                path=self.output_dir / f"seed_{self.seed}_step{step + 1}of{len(self.time_steps)}.png",
                export_json_metadata=False,
            )

    def handle_interruption(self):
        if self.step_wise_images:
            composite_img = ImageUtil.to_composite_image(self.step_wise_images)
            composite_img.save(self.output_dir / f"seed_{self.seed}_composite.png")
