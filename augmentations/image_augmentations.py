import tensorflow as tf
import tensorflow_addons as tfa



class Augs(object):
    def __init__(self):
        self.only_image = True

class OneOf(object):
    def __init__(self, augs):
        self.augs = augs

    def  __call__():
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
            if not issubclass(a, Augs):
                raise TypeError('Augmentations must be subclasses of Augs')
            
            if a.only_image:
                image = a(image)
            else:
                image, mask = a(image, mask)

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

        







class RandomCentralCrop(Augs):

    def __init__(self, min_fraction, max_fraction, p=0.5, seed=None):
        self.__doc__ = tf.image.central_crop.__doc__
        self.only_image = False
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.p = p
        self.seed = seed

    def __call__(self, image, mask):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image, mask

        fraction = tf.random.uniform([], minval=self.min_fraction,
                        maxval=self.max_fraction, dtype=tf.int32, seed=self.seed)

        image = tf.image.central_crop(image, fraction)


        if not mask:
            return image, mask

        mask = tf.image.central_crop(image, fraction)

        return image, mask
     


class RandomRotation(Augs):

    # if tf.random.normal([1], 0, 1) > 0:
    #     rotate_factor = tf.random.normal([1], 32, 8)
    #     neg = tf.constant(np.dtype('float32').type(1))

    #     if tf.random.normal([1], 0, 1) > 0:
    #         neg = tf.constant(np.dtype('float32').type(-1))

    #     rad = (neg * np.pi) / rotate_factor
        
    #     input_image = tfa.image.rotate(input_image, rad)
    #     input_mask = tfa.image.rotate(input_mask, rad)
    

    def __init__(self, rg, fill_mode='nearest', cval=0.0, p=0.5, seed=None):
        self.__doc__ = tf.keras.preprocessing.image.random_rotation.__doc__
        self.only_image = False
        
        self.rg = rg
        self.fill_mode = fill_mode
        self.cval = cval
        self.p = p
        self.seed = seed

    def __call__(self, image, mask):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image, mask

        image = tf.keras.preprocessing.image.random_rotation(
                        image, rg=self.rg, row_axis=0, col_axis=1,
                        channel_axis=2, fill_mode=self.fill_mode, cval=self.cval
                        )

        if not mask:
            return image, mask

        mask = tf.keras.preprocessing.image.random_rotation(
                        mask, rg=self.rg, row_axis=0, col_axis=1,
                        channel_axis=2, fill_mode=self.fill_mode, cval=self.cval
                        )
        return image, mask



class RandomZoom(Augs):

    def __init__(self, zoom_range, fill_mode='nearest', cval=0.0, p=0.5, seed=None):
        self.__doc__ = tf.keras.preprocessing.image.random_zoom.__doc__
        self.only_image = False
        
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.p = p
        self.seed = seed

    def __call__(self, image, mask):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image, mask

        image = tf.keras.preprocessing.image.random_zoom(
                        image, zoom_range=self.zoom_range, row_axis=0, col_axis=1,
                        channel_axis=2, fill_mode=self.fill_mode, cval=self.cval
                        )

        if not mask:
            return image, mask

        mask = tf.keras.preprocessing.image.random_zoom(
                        mask, zoom_range=self.zoom_range, row_axis=0, col_axis=1,
                        channel_axis=2, fill_mode=self.fill_mode, cval=self.cval
                        )
        return image, mask


class RandomShift(Augs):

    def __init__(self, wrg, hrg, fill_mode='nearest', cval=0.0, p=0.5, seed=None):
        self.__doc__ = tf.keras.preprocessing.image.random_zoom.__doc__
        self.only_image = False
        
        self.hrg = hrg
        self.wrg = wrg
        self.fill_mode = fill_mode
        self.cval = cval
        self.p = p
        self.seed = seed

    def __call__(self, image, mask):
        if tf.random.uniform([], dtype=tf.float32, seed=self.seed) > self.p:
            return image, mask

        image = tf.keras.preprocessing.image.random_shift(
            image, wrg=self.wrg, hrg=self.hrg, row_axis=0, col_axis=1,
            channel_axis=2, fill_mode=self.fill_mode, cval=0.0, interpolation_order=1
        )

        if not mask:
            return image, mask

        mask = tf.keras.preprocessing.image.random_shift(
            mask, wrg=self.wrg, hrg=self.hrg, row_axis=0, col_axis=1,
            channel_axis=2, fill_mode=self.fill_mode, cval=0.0, interpolation_order=1
        )
                    
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
        self.wfr = wfraction_range
        self.hfr = hfraction_range
        self.only_image = False
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

        hpad = tf.cast(shape[0] * hpad / 2, tf.int32)
        wpad = tf.cast(shape[1] * wpad / 2, tf.int32)


        image = tf.image.pad_to_bounding_box(
            image, hpad, wpad, shape[0] + hpad * 2, shape[1] + wpad * 2
        )
        
        if not mask:
            return image, mask

        mask = tf.image.pad_to_bounding_box(
            mask, hpad, wpad, shape[0] + hpad * 2, shape[1] + wpad * 2
        )
                    
        return image, mask

