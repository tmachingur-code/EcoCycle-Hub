
from PIL import Image, ImageOps


def prepare_image(image: Image.Image) -> Image.Image:
    """Normalize an uploaded image for model inference."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image.")

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    if image.width < 32 or image.height < 32:
        raise ValueError("Image is too small for reliable analysis.")

    return image


def clip_box(box, image_width: int, image_height: int):
    """Keep bounding-box coordinates within image boundaries."""
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def extract_crop(image: Image.Image, box):
    """Extract an object crop using validated coordinates."""
    coordinates = clip_box(box, image.width, image.height)

    if coordinates is None:
        return None

    return image.crop(coordinates)


def resize_for_display(image: Image.Image, max_size=(1200, 900)):
    """Resize large images while preserving aspect ratio."""
    image = image.copy()
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image
