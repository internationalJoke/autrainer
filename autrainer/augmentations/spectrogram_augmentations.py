from typing import List, Optional
import os
import random

import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms.functional as F

from .abstract_augmentation import AbstractAugmentation
from .spectrogram_warp_utils import _sparse_image_warp


class GaussianNoise(AbstractAugmentation):
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Add Gaussian noise to the input tensor with mean and standard
        deviation.

        Args:
            mean: The mean of the Gaussian noise. Defaults to 0.0.
            std: The standard deviation of the Gaussian noise. Defaults to 1.0.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(order, p, generator_seed)
        self.mean = mean
        self.std = std
        self._generator = torch.Generator()
        if generator_seed is not None:
            self._generator.manual_seed(generator_seed)

    def offset_generator_seed(self, offset: int) -> None:
        super().offset_generator_seed(offset)
        if self.generator_seed is not None:
            self._generator.manual_seed(self.generator_seed)

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        r = torch.randn(x.size(), generator=self._generator)
        return x + r * self.std + self.mean


class StaticGaussianNoise(AbstractAugmentation):
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Add Gaussian noise to the input tensor with mean and standard
        deviation. The noise is deterministic per sample (seeded by index).

        Args:
            mean: The mean of the Gaussian noise. Defaults to 0.0.
            std: The standard deviation of the Gaussian noise. Defaults to 1.0.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(order, p, generator_seed)
        self.mean = mean
        self.std = std

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self.generator_seed + index)
        r = torch.randn(x.size(), generator=generator)
        return x + r * self.std + self.mean


class SNR_noise(AbstractAugmentation):
    """Add Gaussian noise with target SNR to log mel spectrogram.

    The noise is added properly in linear domain:
    1. Convert signal from dB to linear
    2. Generate Gaussian noise with power for target SNR
    3. Add signal + noise in linear domain
    4. Convert back to dB
    """
    available_noise_type = ['Gaussian', 'StaticGaussian']

    def __init__(
        self,
        snr: float = 0.0,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
        noise_type: str = 'Gaussian',
    ) -> None:
        if noise_type not in self.available_noise_type:
            raise ValueError("This noise is not available.")

        super().__init__(order, p, generator_seed)
        self.snr = snr
        self.noise_type = noise_type
        self._generator = torch.Generator()
        if generator_seed is not None:
            self._generator.manual_seed(generator_seed)

    def offset_generator_seed(self, offset: int) -> None:
        super().offset_generator_seed(offset)
        if self.generator_seed is not None:
            self._generator.manual_seed(self.generator_seed)

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        """Apply Gaussian noise with target SNR.

        Properly adds noise in linear domain and converts back to dB.
        """
        # Convert signal from dB to linear domain (power spectrum)
        signal_linear = 10 ** (x / 10)

        # Calculate signal power
        p_signal = signal_linear.mean()

        # Calculate required noise power for target SNR
        # SNR = 10 * log10(P_signal / P_noise)
        # P_noise = P_signal / 10^(SNR/10)
        p_noise_target = p_signal / (10 ** (self.snr / 10))

        # Generate random noise pattern and scale to achieve target power
        if self.noise_type == "Gaussian":
            noise_raw = torch.abs(torch.randn(x.size(), generator=self._generator))
        elif self.noise_type == "StaticGaussian":
            generator = torch.Generator()
            generator.manual_seed(self.generator_seed + index)
            noise_raw = torch.abs(torch.randn(x.size(), generator=generator))
        else:
            return x

        # Scale noise to achieve exactly target mean power
        noise_linear = noise_raw * (p_noise_target / (noise_raw.mean() + 1e-9))

        # Add noise in linear domain
        mixed_linear = signal_linear + noise_linear

        # Convert back to dB
        mixed_db = 10 * torch.log10(mixed_linear + 1e-9)

        return mixed_db


class CrossDomainNoise(AbstractAugmentation):
    """Add real cross-domain noise from AudioSet-Balanced-Noise with target SNR.

    This augmentation loads real noise samples from a directory and adds them
    to the log mel spectrogram with a specified SNR. The noise is:
    1. Loaded as a waveform (.wav)
    2. Converted to log mel spectrogram matching the input's parameters
    3. Cropped or repeated to match the input's time dimension
    4. Scaled to achieve the target SNR
    5. Added to the input spectrogram

    Args:
        noise_dir: Root directory containing noise wav files
        noise_csv: Optional CSV file listing noise files. If None, all .wav files
            in noise_dir will be used
        snr_db: Target Signal-to-Noise Ratio in dB. Lower values = more noise
        sample_rate: Sample rate for loading audio. Defaults to 16000
        n_fft: FFT size for mel spectrogram. Defaults to 512
        hop_length: Hop length for mel spectrogram. Defaults to 160
        n_mels: Number of mel filterbanks. Defaults to 64
        noise_type: Type of noise to use. If specified, filters noise files by
            this label from the CSV. Options: 'Environmental noise', 'Noise',
            'Pink noise', 'White noise'. If None, uses all available noise.
        order: The order of the augmentation in the transformation pipeline
        p: The probability of applying the augmentation. Defaults to 1.0
        generator_seed: The initial seed for the internal random number generator
    """

    def __init__(
        self,
        noise_dir: str,
        snr_db: float,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 64,
        noise_csv: Optional[str] = None,
        noise_type: Optional[str] = None,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        super().__init__(order, p, generator_seed)
        self.noise_dir = noise_dir
        self.snr_db = snr_db
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.noise_type = noise_type

        # Load noise file paths
        self.noise_files = self._load_noise_files(noise_csv)

        if len(self.noise_files) == 0:
            raise ValueError(f"No noise files found in {noise_dir}")

        # Create mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # Cache for loaded noise spectrograms
        self._noise_cache = {}

    def _load_noise_files(self, noise_csv: Optional[str]) -> List[str]:
        """Load list of noise files from directory or CSV."""
        noise_files = []

        if noise_csv is not None and os.path.exists(noise_csv):
            # Load from CSV
            df = pd.read_csv(noise_csv)
            if self.noise_type is not None and 'label' in df.columns:
                # Filter by noise type
                df = df[df['label'] == self.noise_type]

            for path in df['path']:
                full_path = os.path.join(self.noise_dir, path)
                if os.path.exists(full_path):
                    noise_files.append(full_path)
        else:
            # Load all .wav files from directory recursively
            for root, dirs, files in os.walk(self.noise_dir):
                for file in files:
                    if file.endswith('.wav'):
                        noise_files.append(os.path.join(root, file))

        return noise_files

    def _load_noise_spectrogram(self, noise_path: str) -> torch.Tensor:
        """Load a noise file and convert to log mel spectrogram."""
        if noise_path in self._noise_cache:
            return self._noise_cache[noise_path].clone()

        # Load audio
        waveform, sr = torchaudio.load(noise_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Create mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale (dB)
        log_mel_spec = 10 * torch.log10(mel_spec + 1e-9)

        # Cache it
        self._noise_cache[noise_path] = log_mel_spec

        return log_mel_spec.clone()

    def _match_length(self, noise: torch.Tensor, target_length: int) -> torch.Tensor:
        """Crop or repeat noise to match target length."""
        _, _, noise_length = noise.shape

        if noise_length == target_length:
            return noise
        elif noise_length > target_length:
            # Randomly crop
            start = random.randint(0, noise_length - target_length)
            return noise[:, :, start:start + target_length]
        else:
            # Repeat and crop
            repeat_times = (target_length // noise_length) + 1
            noise_repeated = noise.repeat(1, 1, repeat_times)
            return noise_repeated[:, :, :target_length]

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        """Apply cross-domain noise with target SNR.

        The noise is added properly in linear domain:
        1. Convert signal and noise from dB to linear
        2. Scale noise to achieve target SNR
        3. Add signal + noise in linear domain
        4. Convert back to dB
        """
        # Select random noise file
        noise_path = random.choice(self.noise_files)

        # Load noise spectrogram
        noise_spec = self._load_noise_spectrogram(noise_path)

        # Match dimensions to input
        noise_spec = self._match_length(noise_spec, x.shape[-1])

        # Ensure noise has same shape as features
        if noise_spec.shape[1] != x.shape[1]:
            # Resize frequency dimension if needed
            noise_spec = torch.nn.functional.interpolate(
                noise_spec.unsqueeze(0),
                size=(x.shape[1], x.shape[2]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Convert signal from dB to linear domain
        signal_linear = 10 ** (x / 10)

        # Convert noise from dB to linear domain
        noise_linear = 10 ** (noise_spec / 10)

        # Calculate signal power (mean over all dimensions)
        p_signal = signal_linear.mean()

        # Calculate current noise power
        p_noise_current = noise_linear.mean()

        # Calculate required noise power for target SNR
        # SNR = 10 * log10(P_signal / P_noise)
        # P_noise = P_signal / 10^(SNR/10)
        p_noise_target = p_signal / (10 ** (self.snr_db / 10))

        # Scale factor for noise power (no sqrt because we're scaling power, not amplitude)
        # scaled_power = original_power * scale
        noise_scale = p_noise_target / (p_noise_current + 1e-9)

        # Scale noise
        scaled_noise_linear = noise_linear * noise_scale

        # Add signal and noise in linear domain
        mixed_linear = signal_linear + scaled_noise_linear

        # Convert back to dB
        mixed_db = 10 * torch.log10(mixed_linear + 1e-9)

        return mixed_db


class TimeShift(AbstractAugmentation):
    def __init__(
        self,
        axis: int,
        time_steps: int = 0,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Shift the input tensor along the time axis.

        Args:
            axis: Time axis. If the image is torch Tensor, it is expected
                to have [C, H, W] shape, then H is assumed to be axis 0, and W
                is axis 1.
            time_steps: maximum time steps a tensor will shifted
                forward or backward. Defaults to 0.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        if time_steps < 0:
            raise ValueError(f"Time steps '{time_steps}' must be >= 0.")
        super().__init__(order, p, generator_seed)
        self.axis = axis
        self.time_steps = time_steps
        self._generator = torch.Generator()
        if generator_seed is not None:
            self._generator.manual_seed(generator_seed)

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        if self.time_steps == 0:
            return x

        t = torch.randint(
            -self.time_steps, self.time_steps, (1,), generator=self._generator
        ).item()
        if self.axis == 1:
            t1, t2 = x[:, :, :-t].clone(), x[:, :, -t:].clone()
        else:
            t1, t2 = x[:, :-t, :].clone(), x[:, -t:, :].clone()

        return torch.cat((t2, t1), dim=self.axis + 1)


class TimeMask(AbstractAugmentation):
    def __init__(
        self,
        time_mask: int,
        axis: int,
        replace_with_zero: bool = True,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Mask a random number of time steps.

        Important: While the probability of applying the augmentation is
        deterministic if the generator_seed is set, the actual augmentation
        applied is not deterministic. This is because the internal random
        number generator of the augmentation is not seeded.

        Args:
            time_mask: maximum time steps in a tensor will be masked.
            axis: Time axis. If the image is torch Tensor, it is expected
                to have [C, H, W] shape, then H is assumed to be axis 0, and W
                is axis 1.
            replace_with_zero: Fill the mask either with a tensor mean, or 0's.
                Defaults to True.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        if time_mask < 0:
            raise ValueError(f"Time mask '{time_mask}' must be >= 0.")
        super().__init__(order, p, generator_seed)
        self._deterministic = False
        self.time_mask = time_mask
        self.axis = axis
        self.replace_with_zero = replace_with_zero
        self.masking = T.TimeMasking(time_mask_param=self.time_mask)

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        if self.time_mask == 0:
            return x
        if self.axis == 0:
            x = torch.rot90(x, 3, [1, 2])
            x = self.masking(x)
            x = torch.rot90(x, 1, [1, 2])
        else:
            x = self.masking(x)

        return x


class FrequencyMask(AbstractAugmentation):
    def __init__(
        self,
        freq_mask: int,
        axis: int,
        replace_with_zero: bool = True,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Mask a random number of frequency steps.

        Important: While the probability of applying the augmentation is
        deterministic if the generator_seed is set, the actual augmentation
        applied is not deterministic. This is because the internal random
        number generator of the augmentation is not seeded.

        Args:
            freq_mask: maximum frequency steps in a tensor will be masked.
            axis: Frequency axis. If the image is torch Tensor, it is
                expected to have [C, H, W] shape, then H is assumed to be axis 0,
                and W is axis 1.
            replace_with_zero: Fill the mask either with a tensor mean, or 0's.
                Defaults to True.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        if freq_mask < 0:
            raise ValueError(f"Frequency mask '{freq_mask}' must be >= 0.")
        super().__init__(order, p, generator_seed)
        self._deterministic = False
        self.freq_mask = freq_mask
        self.axis = axis
        self.replace_with_zero = replace_with_zero
        self.masking = T.FrequencyMasking(freq_mask_param=self.freq_mask)

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        if self.freq_mask == 0:
            return x
        if self.axis == 1:
            x = torch.rot90(x, 3, [1, 2])
            x = self.masking(x)
            x = torch.rot90(x, 1, [1, 2])
        else:
            x = self.masking(x)

        return x


class TimeWarp(AbstractAugmentation):
    def __init__(
        self,
        axis: int,
        W: int = 10,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """A random point along the time axis passing through the center of
        the image within the time steps (W, tau - W) is to be warped either to
        the left or right by a distance w chosen from a uniform distribution
        from 0 to the time warp parameter W along that line.

        Args:
            axis: Time axis. If the image is torch Tensor, it is expected
                to have [C, H, W] shape, then H is assumed to be axis 0, and W
                is axis 1.
            W: Bound for squishing/stretching. Defaults to 10.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(order, p, generator_seed)
        self.axis = axis
        self.W = W
        self._generator = torch.Generator()
        if generator_seed is not None:
            self._generator.manual_seed(generator_seed)

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        device = x.device
        if self.axis == 0:
            x = torch.rot90(x, 3, [1, 2])

        _, num_freq_channels, len_time = F.get_dimensions(x)

        # random point along the time axis
        pt = (len_time - 2 * self.W) * torch.rand(
            [1],
            dtype=torch.float,
            generator=self._generator,
        ) + self.W

        # source
        # control points on freq-axis
        src_ctr_pt_freq = torch.arange(0, num_freq_channels // 2)
        src_ctr_pt_time = (
            torch.ones_like(src_ctr_pt_freq) * pt
        )  # control points on time-axis
        src_ctr_pts = torch.stack((src_ctr_pt_freq, src_ctr_pt_time), dim=-1)
        src_ctr_pts = src_ctr_pts.float().to(device)

        # Destination
        w = (
            2
            * self.W
            * torch.rand([1], dtype=torch.float, generator=self._generator)
            - self.W
        )  # distance
        dest_ctr_pt_freq = src_ctr_pt_freq
        dest_ctr_pt_time = src_ctr_pt_time + w
        dest_ctr_pts = torch.stack(
            (dest_ctr_pt_freq, dest_ctr_pt_time), dim=-1
        )
        dest_ctr_pts = dest_ctr_pts.float().to(device)

        src_ctr_pt_locations = torch.unsqueeze(src_ctr_pts, 0)
        dest_ctr_pt_locations = torch.unsqueeze(dest_ctr_pts, 0)

        warped_spectro, _ = _sparse_image_warp(
            x, src_ctr_pt_locations, dest_ctr_pt_locations
        )

        if self.axis == 0:
            warped_spectro = torch.rot90(warped_spectro, 1, [1, 2])
        return warped_spectro


class SpecAugment(AbstractAugmentation):
    def __init__(
        self,
        time_mask: int = 10,
        freq_mask: int = 10,
        W: int = 50,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """SpecAugment augmentation.
        A combination of time warp, frequency masking, and time masking.

        Important: While the probability of applying the augmentation is
        deterministic if the generator_seed is set, the actual augmentation
        applied is not deterministic. This is because the internal random
        number generator of the augmentation is not seeded.

        For more information, see:
        https://arxiv.org/abs/1904.08779

        This implementation differs from PyTorch, as they apply TimeStrech
        instead of TimeWarp. For more information, see:
        https://pytorch.org/audio/master/tutorials/audio_feature_augmentation_tutorial.html#specaugment

        Args:
            time_mask: maximum time steps in a tensor will be masked.
                Defaults to 10.
            freq_mask: maximum frequency steps in a tensor will be masked.
                Defaults to 10.
            W: Bound for squishing/stretching the time axis. Defaults to 50.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(order, p, generator_seed)
        self._deterministic = False
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.W = W
        self._time_warp = TimeWarp(W=self.W, axis=0)
        self._freq_mask = FrequencyMask(
            freq_mask=self.freq_mask,
            replace_with_zero=True,
            axis=1,
        )
        self._time_mask = TimeMask(
            time_mask=self.time_mask,
            replace_with_zero=True,
            axis=0,
        )

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        x = self._time_warp(x)
        x = self._freq_mask(x)
        x = self._time_mask(x)
        return x
