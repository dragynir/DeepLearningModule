import tensorflow as tf
import tensorflow_addons as tfa



class Augs(object):
    def __init__(self, only_image):
        self.only_image = only_image
        self.p = 0.5
        self.perform = True

    def random_pick(self):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            self.perform = False
        else:
            self.perform = True

    def build(self):
        self.random_pick()


class OneOf(object):
    def __init__(self, augs):
        self.augs = augs

    def  __call__(self):
        i = tf.random.uniform([], 0, len(self.augs), dtype=tf.int32)
        return self.augs[i]



class Compose(object):

    def __init__(self, augs):
        self.augs = augs

    @tf.function
    def __call__(self, image, mask=None):

        for a in self.augs:
            if isinstance(a, OneOf):
                a = a()
            if not isinstance(a, Augs):
                raise TypeError('Augmentations must be subclasses of Augs')
            
            a.build()

            image = a(image)

            if (not a.only_image and mask):
                mask = a(mask)

        if not mask:
            return image

        return image, mask



# TODO add mean filters



class Equalize(Augs):
    def __init__(self, max_delta, p=0.5, seed=None):
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
            image = tf.cast(tf.add(tf.cast(image, tf.float32), gnoise), tf.uint8)
        return image

        


class RandomRotation(Augs):
    pass


class RandomZoom(Augs):
    pass

class RandomShift(Augs):
    pass



# TODO fix central_crop or rewrite
class RandomCentralCrop(Augs):

    def __init__(self, min_fraction, max_fraction, p=0.5, seed=None):
        super().__init__(only_image=False)
        self.__doc__ = tf.image.central_crop.__doc__
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.p = p
        self.seed = seed

    def build(self):

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

    def build(self):
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