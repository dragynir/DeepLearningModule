import tensorflow as tf
import tensorflow_addons as tfa



class Augs(object):
    def __init__(self, only_image):
        self.only_image = only_image

class OneOf(object):
    def __init__(self, augs):
        self.augs = augs

    def  __call__(self):
        i = tf.random.uniform([], 0, len(self.augs), dtype=tf.int32)
        return self.augs[i]



class Compose(object):

    def __init__(self, augs):
        self.augs = augs

    def __call__(self, image, mask=None):

        for a in self.augs:
            if isinstance(a, OneOf):
                a = a()
            if not isinstance(a, Augs):
                raise TypeError('Augmentations must be subclasses of Augs')
            
            if a.only_image:
                image = a(image)
            else:
                image, mask = a(image, mask)

        if mask is None:
            return image

        return image, mask



# TODO add mean filters



class Equalize(Augs):
    def __init__(self, max_delta, p=0.5, seed=None):
        self.__doc__ = tfa.image.equalize.__doc__
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image
        return tfa.image.equalize(image)


class RandomBrightness(Augs):

    def __init__(self, max_delta, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tf.image.random_brightness.__doc__
        self.max_delta = max_delta
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image
        return tf.image.random_brightness(image, self.max_delta, self.seed)


class RandomHue(Augs):

    def __init__(self, max_delta, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tf.image.random_hue.__doc__
        self.max_delta = max_delta
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image
        return tf.image.random_hue(image, self.max_delta, self.seed)



class RandomSaturation(Augs):

    def __init__(self, lower, upper, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tf.image.random_saturation.__doc__
        self.lower = lower
        self.upper = upper
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image
        return tf.image.random_saturation(image, self.lower, self.upper, self.seed)




class RandomContrast(Augs):

    def __init__(self, lower, upper, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tf.image.random_contrast.__doc__
        self.lower = lower
        self.upper = upper
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image
        return tf.image.random_contrast(image, self.lower, self.upper, self.seed)


class RandomJpegQuality(Augs):

    def __init__(self, min_quality, max_quality, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tf.image.random_contrast.__doc__
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image
        
        quality = tf.random.uniform([], minval=self.min_quality,
                        maxval=self.max_quality, dtype=tf.int32, seed=self.seed)

        return tf.image.adjust_jpeg_quality(image, jpeg_quality=quality)


class GaussianNoise(Augs):

    def __init__(self, mean, stddev, p=0.5, seed=None):
        super().__init__(only_image=True)
        self.__doc__ = tf.random.normal.__doc__
        self.mean = mean
        self.stddev = stddev
        self.p = p
        self.seed = seed

    def __call__(self, image):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image

        gnoise = tf.random.normal(shape=tf.shape(image), mean=self.mean,
                                 stddev=self.stddev, dtype=tf.float32)

        return tf.add(image, gnoise)

        






class RandomRotation(Augs):
    pass



class RandomZoom(Augs):
    pass

class RandomShift(Augs):
    pass




class RandomCentralCrop(Augs):

    def __init__(self, min_fraction, max_fraction, p=0.5, seed=None):
        super().__init__(only_image=False)
        self.__doc__ = tf.image.central_crop.__doc__
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.p = p
        self.seed = seed

    def __call__(self, image, mask):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image, mask

        fraction = tf.random.uniform([], minval=self.min_fraction,
                        maxval=self.max_fraction, dtype=tf.float32, seed=self.seed)

        image = tf.image.central_crop(image, fraction)


        if not mask:
            return image, mask

        mask = tf.image.central_crop(image, fraction)

        return image, mask
     




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

    def __call__(self, image, mask):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image, mask

        wpad = tf.random.uniform([], minval=self.wfr[0],
                                maxval=self.wfr[1], dtype=tf.float32, seed=self.seed)

        hpad = tf.random.uniform([], minval=self.hfr[0],
                                maxval=self.hfr[1], dtype=tf.float32, seed=self.seed)
        
        shape = tf.cast(tf.shape(image), tf.float32)
        hpad = shape[0] * hpad / 2
        wpad = shape[1] * wpad / 2

        end_h, end_w = shape[0] + hpad * 2, shape[1] + wpad * 2

        hpad = tf.cast(hpad, tf.int32)
        wpad = tf.cast(wpad, tf.int32)

        image = tf.image.pad_to_bounding_box(
            image, hpad, wpad, end_h, end_w
        )
        
        if not mask:
            return image, mask

        mask = tf.image.pad_to_bounding_box(
            mask, hpad, wpad, end_h, end_w
        )
                    
        return image, mask

