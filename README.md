# Progressive InfoGAN

Progressive InfoGAN combines two techniques:
- progressive training (https://arxiv.org/abs/1710.10196), and
- InfoGAN (https://arxiv.org/abs/1606.03657).

## What it does
- A Generator is trained on any unlabeled high-res dataset of images. For our experiments we use [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training).
- After training, the Generator produces novel images, similar to those in the dataset.
- When producing an image, we can independently control its semantic feautres. In case of CelebA-HQ, semantic features include: the direction of the look, hair color, nose shape, and much more. See [Summary of selected isolated features](#summary-of-selected-isolated-features).
- Most importantly, these semantic features are discovered during training in a completely unsupervised fashion - no human input is required.

## Under the hood
The InfoGAN technique was adapted to the progressive architecture of the model by splitting the structured code to parts, and feeding each part to the network by conditioning activations in the corresponding block of the Generator. You can find a detailed description of the architecture, along with quality assessment, in my [Master's Thesis text](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/progressive_infogan_draft.pdf) (expected defense: Q4 2018).

## Running
```bash
# Training:
# Fill in src/constants.py, and run
$ cd src
$ python jonasz/experiments/2018_08_28/exp_consistency_300.py

# Working with the trained Generator:
$ cd src
$ python -m jupyter notebook --port 5088
```
Now you can experiment with the generator in a notebook: [src/jonasz/notebooks/generator.ipynb](src/jonasz/notebooks/generator.ipynb)

## Summary of selected isolated features
See also the animated, extended version of the summary at: [https://youtu.be/U2okTa0JGZg](https://youtu.be/U2okTa0JGZg). Some of the subtler changes are better visible when animated. For each of the features listed below, there is a link to the exact corresponding timestamp in the animation.


[Smile: upper lip](#smile-upper-lip-animation02m43s)  
[Age](#age-animation04m22s)  
[Look direction](#look-direction-animation03m16s)  
[Hair color](#hair-color-animation00m06s)  
[Left / right rotation](#left--right-rotation-animation00m15s)  
[Face oval: size](#face-oval-size-animation00m28s)  
[Hairstyle: background size](#hairstyle-background-size-animation01m08s)  
[Lower jaw size](#lower-jaw-size-animation01m51s)  
[Forward backward inclination](#forward--backward-inclination-animation01m41s)  
[Mouth: open / closed](#mouth-open--closed-animation01m31s)  
[Hair: wavy / straight](#hair-wavy--straight-animation02m11s)  
[Eyebrows: up / down](#eyebrows-up--down-animation02m56s)  
[Nose length](#nose-length-animation04m02s)  
[Nose: upturned tip](#nose-upturned-tip-animation03m52s)  
[Eyebrows shape](#eyebrows-shape-animation03m06s)  
[Vertical face stretch](#vertical-face-stretch-animation03m32s)  
[Color of the irises](#color-of-the-irises-animation05m07s)  
[Shape of the nostrils](#shape-of-the-nostrils-animation05m18s)  
[Hair texture](#hair-texture-animation06m10s)  
[Lower eyelid](#lower-eyelid-animation05m54s)  
[Wrinkles](#wrinkles-animation05m27s)  


A comprehensive list of all isolated features is at: [https://youtu.be/mOckeVkM1jU](https://youtu.be/mOckeVkM1jU).


### Smile: upper lip ([animation@02m43s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=02m43s)):
![c_33_smile_upper_lip](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_33_smile_upper_lip.jpg)


### Age ([animation@04m22s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=04m22s)):
![c_53_age](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_53_age.jpg)


### Look direction ([animation@03m16s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=03m16s)):
![c_57_look_direction](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_57_look_direction.jpg)


### Hair color ([animation@00m06s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=00m06s)):
![c_08_hair_color](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_08_hair_color.jpg)


### Left / right rotation ([animation@00m15s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=00m15s)):
![c_03_left_right_rotation](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_03_left_right_rotation.jpg)


### Face oval: size ([animation@00m28s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=00m28s)):
![c_16_face_oval_size](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_16_face_oval_size.jpg)


### Hairstyle: background size ([animation@01m08s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=01m08s)):
![c_24_hairstyle_background_size](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_24_hairstyle_background_size.jpg)


### Lower jaw size ([animation@01m51s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=01m51s)):
![c_32_lower_jaw_size](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_32_lower_jaw_size.jpg)


### Forward / backward inclination ([animation@01m41s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=01m41s)):
![c_43_forward_backward_inclination](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_43_forward_backward_inclination.jpg)


### Mouth: open / closed ([animation@01m31s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=01m31s)):
![c_44_mouth_open_closed](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_44_mouth_open_closed.jpg)


### Hair: wavy / straight ([animation@02m11s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=02m11s)):
![c_45_hair_curly_straight](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_45_hair_curly_straight.jpg)


### Eyebrows: up / down ([animation@02m56s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=02m56s)):
![c_48_eyebrows_up_down](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_48_eyebrows_up_down.jpg)


### Nose length ([animation@04m02s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=04m02s)):
![c_49_nose_length](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_49_nose_length.jpg)


### Nose: upturned tip ([animation@03m52s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=03m52s)):
![c_52_nose_upturned_tip](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_52_nose_upturned_tip.jpg)


### Eyebrows shape ([animation@03m06s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=03m06s)):
![c_55_eyebrows_shape](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_55_eyebrows_shape.jpg)


### Vertical face stretch ([animation@03m32s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=03m32s)):
![c_60_vertical_face_stretch](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_60_vertical_face_stretch.jpg)


### Color of the irises ([animation@05m07s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=05m07s)):
![c_64_color_of_the_irises](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_64_color_of_the_irises.jpg)


### Shape of the nostrils ([animation@05m18s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=05m18s)):
![c_67_shape_of_the_nostrils](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_67_shape_of_the_nostrils.jpg)


### Hair texture ([animation@06m10s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=06m10s)):
![c_68_hair_texture](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_68_hair_texture.jpg)


### Lower eyelid ([animation@05m54s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=05m54s)):
![c_72_lower_eyelid](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_72_lower_eyelid.jpg)


### Wrinkles ([animation@05m27s](https://www.youtube.com/watch?v=U2okTa0JGZg&t=05m27s)):
![c_75_wrinkles](https://raw.githubusercontent.com/jonasz/progressive_infogan/master/data/c_75_wrinkles.jpg)
