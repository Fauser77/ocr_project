import cv2
import typing
import numpy as np
import logging

from . import Image

""" 
Implemented image augmentors:
- RandomBrightness
- RandomErodeDilate
- RandomSharpen
"""

class Augmentor:
    """ Object that should be inherited by all augmentors

    Args:
        random_chance (float, optional): Chance of applying the augmentor. Where 0.0 is never and 1.0 is always. Defaults to 0.5.
        log_level (int, optional): Log level for the augmentor. Defaults to logging.INFO.
    """
    def __init__(self, random_chance: float=0.5, log_level: int = logging.INFO, augment_annotation: bool = False) -> None:
        self._random_chance = random_chance
        self._log_level = log_level
        self._augment_annotation = augment_annotation

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        assert 0 <= self._random_chance <= 1.0, "random chance must be between 0.0 and 1.0"

    def augment(self, data: typing.Union[Image]):
        """ Augment data """
        raise NotImplementedError


class RandomBrightness(Augmentor):
    """ Randomly adjust image brightness """
    def __init__(
        self, 
        random_chance: float = 0.5,
        delta: int = 100,
        log_level: int = logging.INFO,
        augment_annotation: bool = False
        ) -> None:
        """ Randomly adjust image brightness

        Args:
            random_chance (float, optional): Chance of applying the augmentor. Where 0.0 is never and 1.0 is always. Defaults to 0.5.
            delta (int, optional): Integer value for brightness adjustment. Defaults to 100.
            log_level (int, optional): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool, optional): If True, the annotation will be adjusted as well. Defaults to False.
        """
        super(RandomBrightness, self).__init__(random_chance, log_level, augment_annotation)

        assert 0 <= delta <= 255.0, "Delta must be between 0.0 and 255.0"

        self._delta = delta

    def augment(self, image: Image, value: float) -> Image:
        """ Augment image brightness """
        hsv = np.array(image.HSV(), dtype = np.float32)

        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 2] = hsv[:, :, 2] * value

        hsv = np.uint8(np.clip(hsv, 0, 255))

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        image.update(img)

        return image


class RandomErodeDilate(Augmentor):
    """ Randomly erode and dilate image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        kernel_size: typing.Tuple[int, int]=(1, 1), 
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
        ) -> None:
        """ Randomly erode and dilate image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            kernel_size (tuple): Tuple of 2 integers, setting kernel size for erosion and dilation
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Boolean value to determine if annotation should be adjusted. Defaults to False.
        """
        super(RandomErodeDilate, self).__init__(random_chance, log_level, augment_annotation)
        self._kernel_size = kernel_size
        self.kernel = np.ones(self._kernel_size, np.uint8)

    def augment(self, image: Image) -> Image:
        if np.random.rand() <= 0.5:
            img = cv2.erode(image.numpy(), self.kernel, iterations=1)
        else:
            img = cv2.dilate(image.numpy(), self.kernel, iterations=1)

        image.update(img)

        return image

    
class RandomSharpen(Augmentor):
    """ Randomly sharpen image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        alpha: float = 0.25,
        lightness_range: typing.Tuple = (0.75, 2.0),
        kernel: np.ndarray = None,
        kernel_anchor: np.ndarray = None,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
        ) -> None:
        """ Randomly sharpen image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            alpha (float): Float between 0.0 and 1.0 setting bounds for random probability
            lightness_range (tuple): Tuple of 2 floats, setting bounds for random lightness change
            kernel (np.ndarray): Numpy array of kernel for image convolution
            kernel_anchor (np.ndarray): Numpy array of kernel anchor for image convolution
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Boolean to determine if annotation should be augmented. Defaults to False.
        """
        super(RandomSharpen, self).__init__(random_chance, log_level, augment_annotation)

        self._alpha_range = (alpha, 1.0)
        self._ligtness_range = lightness_range
        self._lightness_anchor = 8

        self._kernel = np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1, -1]], dtype=np.float32) if kernel is None else kernel
        self._kernel_anchor = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32) if kernel_anchor is None else kernel_anchor

        assert 0 <= alpha <= 1.0, "Alpha must be between 0.0 and 1.0"

    def augment(self, image: Image) -> Image:
        lightness = np.random.uniform(*self._ligtness_range)
        alpha = np.random.uniform(*self._alpha_range)

        kernel = self._kernel_anchor  * (self._lightness_anchor + lightness) + self._kernel
        kernel -= self._kernel_anchor
        kernel = (1 - alpha) * self._kernel_anchor + alpha * kernel

        # Apply sharpening to each channel
        r, g, b = cv2.split(image.numpy())
        r_sharp = cv2.filter2D(r, -1, kernel)
        g_sharp = cv2.filter2D(g, -1, kernel)
        b_sharp = cv2.filter2D(b, -1, kernel)

        # Merge the sharpened channels back into the original image
        image.update(cv2.merge([r_sharp, g_sharp, b_sharp]))

        return image