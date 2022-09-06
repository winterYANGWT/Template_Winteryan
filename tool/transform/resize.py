import torchvision.transforms.functional as F
import torchvision as tv
from PIL import Image
import torch
import math
import utils

__all__ = [
    'FixedShortestEdgeResize', 'MinimumResize', 'MaximumResize', 'FixedResize'
]


class FixedShortestEdgeResize(object):
    def __init__(
            self,
            shortest_edge_size,
            interpolation=tv.transforms.InterpolationMode.BILINEAR) -> None:
        super().__init__()

        if not isinstance(shortest_edge_size, int):
            msg = f'shortest_edge_size should be int, but got {shortest_edge_size}.'
            raise ValueError(msg)
        else:
            self.shortest_edge_size = shortest_edge_size

        self.interpolation = interpolation

    def __call__(self, image):
        if isinstance(image, Image.Image):
            w = image.width
            h = image.height
        elif isinstance(image, torch.Tensor):
            c, h, w = image.size()
        else:
            msg = 'image shoude be PIL.Image or torch.Tensor, but got {}.'.format(
                type(image))
            raise ValueError(msg)

        if h == w == self.shortest_edge_size:
            return image

        h_scale = h / self.shortest_edge_size
        w_scale = w / self.shortest_edge_size

        if h_scale > w_scale:
            new_w = self.shortest_edge_size
            new_h = math.floor(h / w_scale)
        else:
            new_h = self.shortest_edge_size
            new_w = math.floor(w / h_scale)

        resized_image = F.resize(image,
                                 size=(new_h, new_w),
                                 interpolation=self.interpolation)
        return resized_image


class MinimumResize(object):
    '''
    Resize image and keep the aspect ratio. Make sure that the shortest edge of image is above the minimum_size.
    '''
    def __init__(
            self,
            minimum_size,
            interpolation=tv.transforms.InterpolationMode.BILINEAR) -> None:
        '''
        Args:
            minimum_size(Tuple[int]): The minimum size of resized image and its height and width should be above the minimum size.
            interpolation(torchvision.transforms.InterpolationMode): Desired interpolation method defined in PIL.Image.
        '''
        super().__init__()

        if not isinstance(minimum_size, int):
            msg = 'minimum_size should be int, but got {}.'.format(
                minimum_size)
            raise ValueError(msg)
        else:
            self.minimum_size = minimum_size

        self.interpolation = interpolation

    def __call__(self, image):
        '''
        Args:
            image(Union[PIL.Image.Image, torch.Tensor]): Image to resize and its aspect ratio will be unchanged.
        Returns:
            (Union[PIL.Image.Image, torch.Tensor]): Resized image.
        '''
        if isinstance(image, Image.Image):
            w = image.width
            h = image.height
        elif isinstance(image, torch.Tensor):
            c, h, w = image.size()
        else:
            msg = 'image shoude be PIL.Image or torch.Tensor, but got {}.'.format(
                type(image))
            raise ValueError(msg)

        h_scale = h / self.minimum_size
        w_scale = w / self.minimum_size

        if h_scale >= 1 and w_scale >= 1:
            return image
        else:
            if h_scale > w_scale:
                new_w = self.minimum_size
                new_h = math.floor(1 / w_scale * h)
            elif h_scale < w_scale:
                new_h = self.minimum_size
                new_w = math.floor(1 / h_scale * w)
            else:
                new_h = new_w = self.minimum_size

            resized_image = F.resize(image,
                                     size=(new_h, new_w),
                                     interpolation=self.interpolation)
            return resized_image


class MaximumResize(object):
    '''
    Resize image and keep the aspect ratio. Make sure that the longest edge of image is below the maximum_size.
    '''
    def __init__(
            self,
            maximum_size,
            interpolation=tv.transforms.InterpolationMode.BILINEAR) -> None:
        '''
        Args:
            maximum_size(Tuple[int]): The maximum size of resized image and its height and width should be below the maximum size.
            interpolation(torchvision.transforms.InterpolationMode): Desired interpolation method defined in PIL.Image.
        '''
        super().__init__()

        if not isinstance(maximum_size, int):
            msg = 'maximum_size should be int, but got {}.'.format(
                maximum_size)
            raise ValueError(msg)
        else:
            self.maximum_size = maximum_size

        self.interpolation = interpolation

    def __call__(self, image):
        '''
        Args:
            image(Union[PIL.Image.Image, torch.Tensor]): Image to resize and its aspect ratio will be unchanged.
        Returns:
            (Union[PIL.Image.Image, torch.Tensor]): Resized image.
        '''
        if isinstance(image, Image.Image):
            w = image.width
            h = image.height
        elif isinstance(image, torch.Tensor):
            c, h, w = image.size()
        else:
            msg = 'image shoude be PIL.Image or torch.Tensor, but got {}.'.format(
                type(image))
            raise ValueError(msg)

        h_scale = h / self.maximum_size
        w_scale = w / self.maximum_size

        if h_scale <= 1 and w_scale <= 1:
            return image
        else:
            if h_scale > w_scale:
                new_h = self.maximum_size
                new_w = math.floor(1 / h_scale * w)
            elif h_scale < w_scale:
                new_w = self.maximum_size
                new_h = math.floor(1 / w_scale * h)
            else:
                new_h = new_w = self.maximum_size

            resized_image = F.resize(image,
                                     size=(new_h, new_w),
                                     interpolation=self.interpolation)
            return resized_image


class FixedResize(object):
    def __init__(
            self,
            fixed_size,
            fill_value=0,
            interpolation=tv.transforms.InterpolationMode.BILINEAR) -> None:
        '''
        Resize Image and keep the aspect ratio. The blank area of resize image will be filled with fill_value.

        Args:
            fixed_size(Tuple[int]): The target size of resized image and its shape should be (h,w).
            fill_value(Union[float, Tuple[float]]): The value filled in blank area.
            interpolation(torchvision.transforms.InterpolationMode): Desired interpolation method defined in PIL.Image.
        '''
        super().__init__()

        if not isinstance(fixed_size,
                          tuple) and len(fixed_size) != 2 and not isinstance(
                              (fixed_size[0], fixed_size[1]), int):
            msg = 'fixed_size should be tuple and shape is (h,w), but got {}'.format(
                fixed_size)
            raise ValueError(msg)
        else:
            self.fixed_size = fixed_size

        self.fill_value = fill_value
        self.interpolation = interpolation

    def __call__(self, image):
        '''
        Args:
            image(Union[PIL.Image.Image, torch.Tensor]): Image to resize and its aspect ratio will be unchanged.
        Returns:
            (Union[PIL.Image.Image, torch.Tensor]): Padded image.
        '''
        if isinstance(image, Image.Image):
            w = image.width
            h = image.height
        elif isinstance(image, torch.Tensor):
            c, h, w = image.size()
        else:
            msg = 'image shoude be PIL.Image or torch.Tensor, but got {}.'.format(
                type(image))
            raise ValueError(msg)

        h_scale = h / self.fixed_size[0]
        w_scale = w / self.fixed_size[1]

        if h_scale >= w_scale:
            new_h = self.fixed_size[0]
            new_w = math.floor(1 / h_scale * w)
        else:
            new_w = self.fixed_size[1]
            new_h = math.floor(1 / w_scale * h)

        resized_image = F.resize(image,
                                 size=(new_h, new_w),
                                 interpolation=self.interpolation)

        padding_left, padding_top, padding_right, padding_bottom = utils.calc_padding_size(
            self.fixed_size, (new_h, new_w))
        padded_image = F.pad(
            resized_image,
            padding=[padding_left, padding_top, padding_right, padding_bottom],
            fill=self.fill_value)
        return padded_image
