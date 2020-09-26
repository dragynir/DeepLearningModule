import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


# TODO move p and seed to Augs
# TODO add seed to some classes

class Augs(object):
    def __init__(self, only_image):
        self.only_image = only_image
        self.p = 0.5
        self.perform = True
        self.seed = None

    def random_pick(self):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            self.perform = False
        else:
            self.perform = True

    def build(self, shape):
        self.random_pick()


class OneOf(object):
    def __init__(self, augs):
        self.augs = augs

    def  __call__(self):
        i = np.random.randint(0, len(self.augs))
        return self.augs[i] 


class SegmCompose(object):

    def __init__(self, augs):
        self.augs = augs

    @tf.function
    def __call__(self, image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:

        for a in self.augs:
            if isinstance(a, OneOf):
                a = a()
            if not isinstance(a, Augs):
                raise TypeError('Augmentations must be subclasses of Augs')
            
            a.build(tf.shape(image))
            image = a(image)

            if not a.only_image:
                mask = a(mask)

        return image, mask


class Compose(object):
    def __init__(self, augs):
        self.augs = augs

    @tf.function
    def __call__(self, image: tf.Tensor) -> tf.Tensor:

        for a in self.augs:
            if isinstance(a, OneOf):
                a = a()
            if not isinstance(a, Augs):
                raise TypeError('Augmentations must be subclasses of Augs')
            
            a.build(tf.shape(image))
            image = a(image)

        return image




# TODO add mean filters




# TODO check operation(None type)
class Equalize(Augs):
    def __init__(self, max_delta, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tfa.image.equalize.__doc__
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if self.perform:
            image = tfa.image.equalize(image)
        return image


class RandomBrightness(Augs):

    def __init__(self, max_delta, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tf.image.random_brightness.__doc__
        self.max_delta = max_delta
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if self.perform:
            image = tf.image.random_brightness(image, self.max_delta, self.seed)
        return image


class RandomHue(Augs):

    def __init__(self, max_delta, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tf.image.random_hue.__doc__
        self.max_delta = max_delta
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if self.perform:
            image = tf.image.random_hue(image, self.max_delta, self.seed)

        return image



class RandomSaturation(Augs):

    def __init__(self, lower, upper, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tf.image.random_saturation.__doc__
        self.lower = lower
        self.upper = upper
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if self.perform:
            image = tf.image.random_saturation(image, self.lower, self.upper, self.seed)

        return image




class RandomContrast(Augs):

    def __init__(self, lower, upper, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tf.image.random_contrast.__doc__
        self.lower = lower
        self.upper = upper
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if self.perform:
            image = tf.image.random_contrast(image, self.lower, self.upper, self.seed)

        return image


class RandomJpegQuality(Augs):

    def __init__(self, min_quality, max_quality, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tf.image.random_contrast.__doc__
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if self.perform:
            quality = tf.random.uniform([], minval=self.min_quality,
                            maxval=self.max_quality, dtype=tf.int32, seed=self.seed)
            image = tf.image.adjust_jpeg_quality(image, jpeg_quality=quality)

        return image



class GaussianNoise(Augs):

    def __init__(self, mean, stddev, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tf.random.normal.__doc__
        self.mean = mean
        self.stddev = stddev
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if self.perform:
            gnoise = tf.random.normal(shape=tf.shape(image), mean=self.mean,
                                 stddev=self.stddev, dtype=tf.float32)
            image = tf.clip_by_value(tf.add(tf.cast(image, tf.float32), gnoise),
                            clip_value_min=0.0, clip_value_max=255.0)
            image = tf.cast(image, tf.uint8)
        return image

        




class RandomShift(Augs):
    pass


class RandomSharpness(Augs):
    pass


class ChannelsShift(Augs):
    pass








# decode image
class RandomFdaTransfer(Augs):

    '''
        target_images: array of images (H, W, C)
    '''

    def __init__(self, target_images, transfer_size, l_range=(0.01, 0.05), p=0.5, seed=None):
        super().__init__(only_image=True)
        self.transfer_size = transfer_size
        self.targets_count = len(target_images)
        self.target_images = tf.stack(target_images)
        self.l_range = l_range
        self.p = p
        self.seed = seed

    @staticmethod
    def __transfer_amp(amp_src, amp_trg, L):

        src = tf.signal.fftshift(amp_src, axes=[0, 1])
        trg = tf.signal.fftshift(amp_trg, axes=[0, 1])

        shape = tf.cast(tf.shape(src), tf.float32)

        h = shape[0]
        w = shape[1]

        b = tf.math.floor(tf.math.reduce_min((h, w)) * L)

        # compute image center
        c_h = tf.math.floor(h/2.0)
        c_w = tf.math.floor(w/2.0)

        # compute central square to transfer
        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1

        start_w = tf.cast(w1, tf.int32)
        len_w = tf.cast(w2 - w1, tf.int32)

        start_h = tf.cast(h1, tf.int32)
        len_h = tf.cast(h2 - h1, tf.int32)


        w = tf.cast(w, tf.int32)
        h = tf.cast(h, tf.int32)

        trg_slice = tf.slice(trg,
                [start_h, start_w, 0], [len_h, len_w, -1])
            

        center_mask = tf.ones_like(trg_slice)

        mask_canvas = tf.image.pad_to_bounding_box(
            [center_mask],
            start_h,
            start_w,
            h,
            w
        )

        bool_mask = tf.squeeze(tf.cast(mask_canvas, tf.bool))

        src_with_trg = tf.where(bool_mask, trg, src)

        return tf.signal.ifftshift(src_with_trg, axes=[0, 1])

    @staticmethod
    def transfer_domain(src_image, trg_image, L):

        src_image = tf.cast(src_image, tf.complex64)
        trg_image = tf.cast(trg_image, tf.complex64)

        src_fft = tf.signal.fft3d(src_image)
        trg_fft = tf.signal.fft3d(trg_image)

        tr_amp = RandomFdaTransfer.__transfer_amp(tf.abs(src_fft), tf.abs(trg_fft), L=L)
        tr_amp = tf.cast(tr_amp, tf.complex64)

        src_fft_mutated = tr_amp * tf.math.exp(tf.cast(tf.math.angle(src_fft),
                                      tf.complex64) * tf.complex([0.0], [1.0]))

        return tf.math.real(tf.signal.ifft3d(src_fft_mutated))


    def __call__(self, image):
        if self.perform:
            image = tf.image.resize(image, self.transfer_size)

            target_ind = tf.random.uniform([], minval=0,
                    maxval=self.targets_count, dtype=tf.int32)

            # write load and decode image
            trg_image = self.target_images[target_ind]

            trg_image = tf.image.resize(trg_image, self.transfer_size)

            L = tf.random.uniform([], minval=self.l_range[0],
                    maxval=self.l_range[1])

            image = RandomFdaTransfer.transfer_domain(image, trg_image, L)

            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

            image = tf.cast(image, tf.uint8)

            image = tf.image.resize(image, self.transfer_size)

        return image


class RandomZoom(Augs):
    
    def __init__(self, target_size, zoom_max=0.8, count=10, p=0.5, seed=None):
        super().__init__(only_image=False)
        self.target_size = target_size
        self.p = p
        self.seed = seed
        self.__generate_boxes(zoom_max, count)
        

    def __generate_boxes(self, zoom_max, count):
        step = (1.0 - zoom_max) / count
        scales = list(np.arange(zoom_max, 1.0, step))
        bx = np.zeros((len(scales), 4))
        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            bx[i] = [x1, y1, x2, y2]
        
        self.boxes_count = len(bx)
        self.boxes = tf.constant(bx, dtype=tf.float32)

    def build(self, shape):

        self.pick_i = tf.random.uniform([], minval=0,
                maxval=self.boxes_count, dtype=tf.int32)
        
        self.random_pick()

    def __call__(self, image):

        if self.perform:
            box = tf.reshape(self.boxes[self.pick_i], (1, 4))

            image = tf.image.crop_and_resize([image], boxes=box,
                box_indices=[0], crop_size=self.target_size)
            
            image = tf.cast(image, tf.uint8)
            image = tf.squeeze(image)
        return image





class RandomRotate90(Augs):
    def __init__(self, k_range = (0, 4), p=0.5, seed=None):
        super().__init__(only_image=False)
        self.k_range = k_range
        self.p = p
        self.seed = seed

    def build(self, shape):

        self.k = tf.random.uniform([], minval=self.k_range[0],
                maxval=self.k_range[1], dtype=tf.int32)
        
        self.random_pick()

    def __call__(self, image):
        if self.perform:
            image = tf.image.rot90(image, k=self.k)
        return image


class FlipLeftRight(Augs):
    def __init__(self, p=0.5, seed=None):
        super().__init__(only_image=False)
        self.__doc__ = tf.image.flip_left_right.__doc__
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if self.perform:
            image = tf.image.flip_left_right(image)
        return image
        

class FlipUpDown(Augs):
    def __init__(self, p=0.5, seed=None):
        super().__init__(only_image=False)
        self.__doc__ = tf.image.flip_up_down.__doc__
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if self.perform:
            image = tf.image.flip_up_down(image)
        return image


class RandomCrop(Augs):

    def __init__(self, width, height, p=0.5, seed=None):
        super().__init__(only_image=False)
        self.width = width
        self.height = height
        self.p = p
        self.seed = seed

    
    def build(self, shape):

        max_width_step = shape[0] - self.width
        max_height_step = shape[1] - self.height

        tf.debugging.Assert(tf.math.less(-1, max_height_step),
        ["RandomCrop: Crop height is bigger then image height", max_height_step])

        tf.debugging.Assert(tf.math.less(-1, max_width_step),
        ["RandomCrop: Crop width is bigger then image width", max_width_step]) 

        self.w_step = tf.random.uniform([], minval=0, dtype=tf.int32,
                        maxval=max_width_step, seed=self.seed)

        self.h_step = tf.random.uniform([], minval=0, dtype=tf.int32,
                        maxval=max_height_step, seed=self.seed)

        self.random_pick()
    
    def __call__(self, image):
        if self.perform:
            image = tf.slice(image,[self.w_step, self.h_step, 0],
                    [self.width, self.height, -1])

        return image





class RandomRotation(Augs):
    

    def __init__(self, angle_range=(-10, 10), interpolation='BILINEAR', p=0.5, seed=None):
        super().__init__(only_image=False)
        self.__doc__ = tfa.image.rotate.__doc__
        self.angle_range = angle_range
        self.interpolation = interpolation
        self.p = p
        self.pi = tf.constant(np.pi)
        self.seed = seed
 
    def build(self, shape):
        angle = tf.random.uniform([], minval=self.angle_range[0],
                                     maxval=self.angle_range[1], dtype=tf.int32)

        self.rad = tf.cast(angle, tf.float32) * self.pi / 180.0

        self.random_pick()

    def __call__(self, image):
        if self.perform:
            image = tfa.image.rotate(image, self.rad, self.interpolation)

        return image



class RandomCentralCrop(Augs):

    def __init__(self, min_fraction, max_fraction, p=0.5, seed=None):
        super().__init__(only_image=False)
        self.__doc__ = tf.image.central_crop.__doc__
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.p = p
        self.seed = seed

    def build(self, shape):

        self.fraction = tf.random.uniform([], minval=self.min_fraction,
                        maxval=self.max_fraction, dtype=tf.float32, seed=self.seed)

        self.random_pick()

    def __call__(self, image):
        if self.perform:

            shape = tf.cast(tf.shape(image), tf.float32)

            width_size = shape[0] * self.fraction
            height_size = shape[1] * self.fraction

            width_r = tf.cast((shape[0] - width_size) / 2.0, tf.int32)
            height_r = tf.cast((shape[1] - height_size) / 2.0, tf.int32)
            

            width_size = tf.cast(width_size, tf.int32)
            height_size = tf.cast(height_size, tf.int32)
            image = tf.slice(image, [width_r, height_r, 0], [width_size, height_size, -1])

        return image



class RandomPad(Augs):
    ''''
        Pad image with same size in horizontal | vertical 
        For example: if pad widht=0.5 then 
        pad left with 0.25 * image_width,
        pad right with 0.25 * image_height
    
        Arguments:
            wfraction_range - list: (min_pad_width, max_pad_width)
            hfraction_range - list: (min_pad_height, max_pad_height)
    
    '''

    def __init__(self, wfraction_range, hfraction_range, p=0.5, seed=None):
        super().__init__(only_image=False)
        self.wfr = wfraction_range
        self.hfr = hfraction_range
        self.p = p
        self.seed = seed
        self.perform = True

    def build(self, shape):
        self.wpad = tf.random.uniform([], minval=self.wfr[0],
                                maxval=self.wfr[1], dtype=tf.float32, seed=self.seed)

        self.hpad = tf.random.uniform([], minval=self.hfr[0],
                                maxval=self.hfr[1], dtype=tf.float32, seed=self.seed)

        self.random_pick()
        

    def __call__(self, image):

        shape = tf.cast(tf.shape(image), tf.float32)
        hpad = shape[0] * self.hpad / 2
        wpad = shape[1] * self.wpad / 2
        end_h, end_w = shape[0] + hpad * 2.0, shape[1] + wpad * 2.0
        
        hpad = tf.cast(hpad, tf.int32)
        wpad = tf.cast(wpad, tf.int32)
        end_w = tf.cast(end_w, tf.int32)
        end_h = tf.cast(end_h, tf.int32)

        if self.perform:
            image = tf.image.pad_to_bounding_box(
                image, hpad, wpad, end_h, end_w
            )
        return image