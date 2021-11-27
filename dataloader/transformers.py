import cv2
class Rescale():
    """Zoom the size of the picture to the specified size

    parameter:
        output_size (width, height) (tuple)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, sample):
        sample = cv2.resize(sample, dsize=self.output_size, interpolation=cv2.INTER_NEAREST)

        return sample
