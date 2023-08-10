from django.shortcuts import render
from django.conf import settings
# Create your views here. 
import Deforum_Stable_Diffusion 
import torch
import random
import clip
from IPython import display
from types import SimpleNamespace
from helpers.save_images import get_output_folder
from helpers.settings import load_args
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import make_linear_decode, load_model, get_model_output_paths
from helpers.aesthetics import load_aesthetics_model
from helpers.prompts import Prompts
import subprocess, time, gc, os, sys

from rest_framework import viewsets
from django.http import JsonResponse
from PIL import Image
from django.core.files.storage import FileSystemStorage
import json
import cv2
import os
drive_mounted = os.path.exists('/content/drive')
class Generate(viewsets.ModelViewSet):        
    def create(self, request):
        prompts = request.data['prompts']
        neg_prompts = """Pornography and explicit adult content,
        Nudity and sexually suggestive images,
        Graphic violence and gore,
        Hate symbols and offensive imagery,
        Child exploitation and abuse-related content,
        Racially offensive or discriminatory images,
        Images promoting terrorism or extremist ideologies,
        Animal cruelty and harm-related images,
        Self-harm or suicide-related images,
        Images promoting illegal activities or substances,
        Sensitive medical imagery and private information,
        Misleading or deceptive images,
        Images containing copyrighted material without permission,
        Violent or abusive images targeting individuals or groups,
        Images that promote fraud or scam activities,
        Harmful images related to weapons or dangerous objects,
        Images containing personal identification information (PII) without consent,
        Images depicting harmful stereotypes or derogatory portrayals of individuals or communities,
        """
        neg_prompts += request.data['neg_prompts']
        image_size = request.data['image_size']
        use_init_state = False
        animode = None
        skip_video_state = True
        total_frames = 20
        video_path = None
        fps = 10
        if 'mode' in request.data:
            dict_ = json.loads(prompts)
            dict_ = {int(key): value for key, value in dict_.items()}
            last_key = list(dict_.keys())[-1]
            prompts = dict_
            total_frames = last_key + 20
            skip_video_state = False
            if request.data['mode'] == '2D':
                animode = '2D'
            elif request.data['mode'] == '3D':
                animode = '3D'
            elif request.data['mode'] == 'Interpolation':
                animode = 'Interpolation'
            else:
                animode = None
        
        init_image_path = "None"
        if image_size == 'Square':
            width = 512
            height = 512
        elif image_size == 'Landscape':
            width = 854
            height = 480
        elif image_size == 'Portrait':
            width = 480
            height = 854
        else:
            width = 512
            height = 512
        if 'image' in request.FILES:            
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            input_file_name = fs.save(uploaded_file.name, uploaded_file)
            uploaded_url = fs.url(input_file_name)
            use_init_state = True
            if drive_mounted:
                init_image_path = '/content/drive/MyDrive/deforum-stable-diffusion-ui/' + uploaded_url
            else:
                init_image_path = '/home/user/Projects/deforum/' + uploaded_url
        if 'video' in request.FILES:     
            skip_video_state = False
            animode = 'Video Input'
            uploaded_file = request.FILES['video']
            fs = FileSystemStorage()
            input_file_name = fs.save(uploaded_file.name, uploaded_file)
            uploaded_url = fs.url(input_file_name)
            if drive_mounted:
                video_path = '/content/drive/MyDrive/deforum-stable-diffusion-ui/' + uploaded_url
            else:
                video_path = '/home/user/Projects/deforum/' + uploaded_url
            fps = Generate.get_video_fps(video_path)
        def DeforumAnimArgs():

            #@markdown ####**Animation:**
            animation_mode = animode #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
            max_frames = total_frames #@param {type:"number"}
            border = 'replicate' #@param ['wrap', 'replicate'] {type:'string'}

            #@markdown ####**Motion Parameters:**
            angle = "0:(0)"#@param {type:"string"}
            zoom = "0:(1)"#@param {type:"string"}
            translation_x = "0:(0)"#@param {type:"string"}
            translation_y = "0:(0)"#@param {type:"string"}
            translation_z = "0:(10)"#@param {type:"string"}
            rotation_3d_x = "0:(0)"#@param {type:"string"}
            rotation_3d_y = "0:(0)"#@param {type:"string"}
            rotation_3d_z = "0:(0)"#@param {type:"string"}
            flip_2d_perspective = False #@param {type:"boolean"}
            perspective_flip_theta = "0:(0)"#@param {type:"string"}
            perspective_flip_phi = "0:(t%15)"#@param {type:"string"}
            perspective_flip_gamma = "0:(0)"#@param {type:"string"}
            perspective_flip_fv = "0:(53)"#@param {type:"string"}
            noise_schedule = "0: (0.02)"#@param {type:"string"}
            strength_schedule = "0: (0.65)"#@param {type:"string"}
            contrast_schedule = "0: (1.0)"#@param {type:"string"}
            hybrid_comp_alpha_schedule = "0:(1)" #@param {type:"string"}
            hybrid_comp_mask_blend_alpha_schedule = "0:(0.5)" #@param {type:"string"}
            hybrid_comp_mask_contrast_schedule = "0:(1)" #@param {type:"string"}
            hybrid_comp_mask_auto_contrast_cutoff_high_schedule =  "0:(100)" #@param {type:"string"}
            hybrid_comp_mask_auto_contrast_cutoff_low_schedule =  "0:(0)" #@param {type:"string"}

            #@markdown ####**Sampler Scheduling:**
            enable_schedule_samplers = False #@param {type:"boolean"}
            sampler_schedule = "0:('euler'),10:('dpm2'),20:('dpm2_ancestral'),30:('heun'),40:('euler'),50:('euler_ancestral'),60:('dpm_fast'),70:('dpm_adaptive'),80:('dpmpp_2s_a'),90:('dpmpp_2m')" #@param {type:"string"}

            #@markdown ####**Unsharp mask (anti-blur) Parameters:**
            kernel_schedule = "0: (5)"#@param {type:"string"}
            sigma_schedule = "0: (1.0)"#@param {type:"string"}
            amount_schedule = "0: (0.2)"#@param {type:"string"}
            threshold_schedule = "0: (0.0)"#@param {type:"string"}

            #@markdown ####**Coherence:**
            color_coherence = 'Match Frame 0 LAB' #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB', 'Video Input'] {type:'string'}
            color_coherence_video_every_N_frames = 1 #@param {type:"integer"}
            color_force_grayscale = False #@param {type:"boolean"}
            diffusion_cadence = '1' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}

            #@markdown ####**3D Depth Warping:**
            use_depth_warping = True #@param {type:"boolean"}
            midas_weight = 0.9 #@param {type:"number"}
            near_plane = 200
            far_plane = 10000
            fov = 40#@param {type:"number"}
            padding_mode = 'border'#@param ['border', 'reflection', 'zeros'] {type:'string'}
            sampling_mode = 'bicubic'#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
            save_depth_maps = False #@param {type:"boolean"}

            #@markdown ####**Video Input:**
            video_init_path = video_path #@param {type:"string"}
            extract_nth_frame = 1 #@param {type:"number"}
            overwrite_extracted_frames = True #@param {type:"boolean"}
            use_mask_video = True #@param {type:"boolean"}
            video_mask_path = video_path#@param {type:"string"}

            #@markdown ####**Hybrid Video for 2D/3D Animation Mode:*    *
            hybrid_generate_inputframes = False #@param {type:"boolean"}
            hybrid_use_first_frame_as_init_image = False #@param {type:"boolean"}
            hybrid_motion = "None" #@param ['None','Optical Flow','Perspective','Affine']
            hybrid_motion_use_prev_img = False #@param {type:"boolean"}
            hybrid_flow_method = "RAFT" #@param ['DenseRLOF','DIS Medium','Farneback','SF']
            hybrid_composite = False #@param {type:"boolean"}
            hybrid_comp_mask_type = "None" #@param ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
            hybrid_comp_mask_inverse = False #@param {type:"boolean"}
            hybrid_comp_mask_equalize = "None" #@param  ['None','Before','After','Both']
            hybrid_comp_mask_auto_contrast = False #@param {type:"boolean"}
            hybrid_comp_save_extra_frames = False #@param {type:"boolean"}
            hybrid_use_video_as_mse_image = False #@param {type:"boolean"}

            #@markdown ####**Interpolation:**
            interpolate_key_frames = True #@param {type:"boolean"}
            interpolate_x_frames = total_frames #@param {type:"number"}
            
            #@markdown ####**Resume Animation:**
            resume_from_timestring = False #@param {type:"boolean"}
            resume_timestring = "20220829210106" #@param {type:"string"}

            return locals()



        #@markdown **Load Settings**
        override_settings_with_file = False #@param {type:"boolean"}
        settings_file = "custom" #@param ["custom", "512x512_aesthetic_0.json","512x512_aesthetic_1.json","512x512_colormatch_0.json","512x512_colormatch_1.json","512x512_colormatch_2.json","512x512_colormatch_3.json"]
        custom_settings_file = "/content/drive/MyDrive/Settings.txt"#@param {type:"string"}
        
        def DeforumArgs():

            #@markdown **Image Settings**
            W = 1920 #@param
            H = 1080 #@param
            W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
            bit_depth_output = 8 #@param [8, 16, 32] {type:"raw"}

            #@markdown **Sampling Settings**
            seed = -1 #@param
            sampler = 'ddim' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
            steps = 200 #@param
            scale = 10 #@param
            ddim_eta = 0.0 #@param
            dynamic_threshold = None
            static_threshold = None   

            #@markdown **Save & Display Settings**
            save_samples = True #@param {type:"boolean"}
            save_settings = True #@param {type:"boolean"}
            display_samples = True #@param {type:"boolean"}
            save_sample_per_step = False #@param {type:"boolean"}
            show_sample_per_step = False #@param {type:"boolean"}

            #@markdown **Batch Settings**
            n_batch = 1 #@param
            n_samples = 1 #@param
            batch_name = "StableFun" #@param {type:"string"}
            filename_format = "{timestring}_{index}_{prompt}.png" #@param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
            seed_behavior = "iter" #@param ["iter","fixed","random","ladder","alternate"]
            seed_iter_N = 1 #@param {type:'integer'}
            make_grid = False #@param {type:"boolean"}
            grid_rows = 2 #@param 
            outdir = get_output_folder(settings.ROOT_VAR.output_path, batch_name)

            #@markdown **Init Settings**
            use_init = use_init_state #@param {type:"boolean"}
            strength = 0.5   #@param {type:"number"}'
            strength_0_no_init = True # Set the strength to 0 automatically when no init image is used
            init_image = init_image_path #@param {type:"string"}
            add_init_noise = True #@param {type:"boolean"}
            init_noise = 0.01 #@param
            # Whiter areas of the mask are areas that change more
            use_mask = False #@param {type:"boolean"}
            use_alpha_as_mask = False # use the alpha channel of the init image as the mask
            mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg" #@param {type:"string"}
            invert_mask = False #@param {type:"boolean"}
            # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
            mask_brightness_adjust = 1.0  #@param {type:"number"}
            mask_contrast_adjust = 1.0  #@param {type:"number"}
            # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
            overlay_mask = True  # {type:"boolean"}
            # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
            mask_overlay_blur = 5 # {type:"number"}

            #@markdown **Exposure/Contrast Conditional Settings**
            mean_scale = 0 #@param {type:"number"}
            var_scale = 0 #@param {type:"number"}
            exposure_scale = 0 #@param {type:"number"}
            exposure_target = 0.5 #@param {type:"number"}

            #@markdown **Color Match Conditional Settings**
            colormatch_scale = 0 #@param {type:"number"}
            colormatch_image = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png" #@param {type:"string"}
            colormatch_n_colors = 4 #@param {type:"number"}
            ignore_sat_weight = 0 #@param {type:"number"}

            #@markdown **CLIP\Aesthetics Conditional Settings**
            clip_name = 'ViT-L/14' #@param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
            clip_scale = 0 #@param {type:"number"}
            aesthetics_scale = 0 #@param {type:"number"}
            cutn = 1 #@param {type:"number"}
            cut_pow = 0.0001 #@param {type:"number"}

            #@markdown **Other Conditional Settings**
            init_mse_scale = 0 #@param {type:"number"}
            init_mse_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg" #@param {type:"string"}
            blue_scale = 0 #@param {type:"number"}
            
            #@markdown **Conditional Gradient Settings**
            gradient_wrt = 'x0_pred' #@param ["x", "x0_pred"]
            gradient_add_to = 'both' #@param ["cond", "uncond", "both"]
            decode_method = 'linear' #@param ["autoencoder","linear"]
            grad_threshold_type = 'dynamic' #@param ["dynamic", "static", "mean", "schedule"]
            clamp_grad_threshold = 0.2 #@param {type:"number"}
            clamp_start = 0.2 #@param
            clamp_stop = 0.01 #@param
            grad_inject_timing = list(range(1,10)) #@param

            #@markdown **Speed vs VRAM Settings**
            cond_uncond_sync = True #@param {type:"boolean"}
            precision = 'autocast' 
            C = 4
            f = 8

            cond_prompt = ""
            cond_prompts = ""
            uncond_prompt = ""
            uncond_prompts = ""
            timestring = ""
            init_latent = None
            init_sample = None
            init_sample_raw = None
            mask_sample = None
            init_c = None
            seed_internal = 0

            return locals()

        args_dict = DeforumArgs()
        anim_args_dict = DeforumAnimArgs()

        if override_settings_with_file:
            load_args(args_dict, anim_args_dict, settings_file, custom_settings_file, verbose=False)

        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)

        args.timestring = time.strftime('%Y%m%d%H%M%S')
        args.strength = max(0.0, min(1.0, args.strength))

        # Load clip model if using clip guidance
        if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
            settings.ROOT_VAR.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(settings.ROOT_VAR.device)
            if (args.aesthetics_scale > 0):
                settings.ROOT_VAR.aesthetics_model = load_aesthetics_model(args, settings.ROOT_VAR)

        if args.seed == -1:
            args.seed = random.randint(0, 2**32 - 1)
        if not args.use_init:
            args.init_image = None
        if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
            print(f"Init images aren't supported with PLMS yet, switching to KLMS")
            args.sampler = 'klms'
        if args.sampler != 'ddim':
            args.ddim_eta = 0

        if anim_args.animation_mode == 'None':
            anim_args.max_frames = 1
        elif anim_args.animation_mode == 'Video Input':
            args.use_init = True

        # clean up unused memory
        gc.collect()
        torch.cuda.empty_cache()

        # get prompts
        cond, uncond = Prompts(prompt=prompts,neg_prompt=neg_prompts).as_dict()

        # dispatch to appropriate renderer
        if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
            render_animation(settings.ROOT_VAR, anim_args, args, cond, uncond)
        elif anim_args.animation_mode == 'Video Input':
            render_input_video(settings.ROOT_VAR, anim_args, args, cond, uncond)
        elif anim_args.animation_mode == 'Interpolation':
            render_interpolation(settings.ROOT_VAR, anim_args, args, cond, uncond)
        else:
            input_file_name = render_image_batch(settings.ROOT_VAR, args, cond, uncond)


        skip_video_for_run_all = skip_video_state #@param {type: 'boolean'}
        create_gif = False #@param {type: 'boolean'}

        if skip_video_for_run_all == True:
            print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
        else:

            from helpers.ffmpeg_helpers import get_extension_maxframes, get_auto_outdir_timestring, get_ffmpeg_path, make_mp4_ffmpeg, make_gif_ffmpeg, patrol_cycle

            def ffmpegArgs():
                ffmpeg_mode = "auto" #@param ["auto","manual","timestring"]
                ffmpeg_outdir = "" #@param {type:"string"}
                ffmpeg_timestring = "" #@param {type:"string"}
                ffmpeg_image_path = "" #@param {type:"string"}
                ffmpeg_mp4_path = "" #@param {type:"string"}
                ffmpeg_gif_path = "" #@param {type:"string"}
                ffmpeg_extension = "png" #@param {type:"string"}
                ffmpeg_maxframes = 200 #@param
                ffmpeg_fps = fps #@param

                # determine auto paths
                if ffmpeg_mode == 'auto':
                    ffmpeg_outdir, ffmpeg_timestring = get_auto_outdir_timestring(args,ffmpeg_mode)
                if ffmpeg_mode in ["auto","timestring"]:
                    ffmpeg_extension, ffmpeg_maxframes = get_extension_maxframes(args,ffmpeg_outdir,ffmpeg_timestring)
                    ffmpeg_image_path, ffmpeg_mp4_path, ffmpeg_gif_path = get_ffmpeg_path(ffmpeg_outdir, ffmpeg_timestring, ffmpeg_extension)
                return locals()

            ffmpeg_args_dict = ffmpegArgs()
            ffmpeg_args = SimpleNamespace(**ffmpeg_args_dict)
            make_mp4_ffmpeg(ffmpeg_args, display_ffmpeg=True, debug=False)
            if create_gif:
                make_gif_ffmpeg(ffmpeg_args, debug=False)
            #patrol_cycle(args,ffmpeg_args)



        skip_disconnect_for_run_all = True #@param {type: 'boolean'}

        if skip_disconnect_for_run_all == True:
            print('Skipping disconnect, uncheck skip_disconnect_for_run_all if you want to run it')
        else:
            from google.colab import runtime
            runtime.unassign()

        if drive_mounted:
            prefix_to_remove = "/content/drive/MyDrive/deforum-stable-diffusion-ui/"
        else:
            prefix_to_remove = "/home/user/Projects/deforum/"
        if skip_video_state == False:
            input_file_name = ffmpeg_args.ffmpeg_mp4_path   
            path = input_file_name.split(prefix_to_remove, 1)[-1]
        else:
            result_string = args.outdir.split(prefix_to_remove, 1)[-1]
            path = result_string + '/' + input_file_name
        data = {
            "path": path
        }


        input_path = path
        output_path = path 

        if image_size == 'Square':
            target_size = (640, 640)
        elif image_size == 'Landscape':
            target_size = (1280, 720)
        elif image_size == 'Portrait':
            target_size = (720,1280)
        else:
            target_size = (640, 640)
        if skip_video_state == True and video_path == None:
            Generate.resize_image(input_path, output_path, target_size)
            if os.path.exists(init_image_path):
                os.remove(init_image_path)
        if skip_video_state == False and video_path != None:
            if os.path.exists(video_path):
                os.remove(video_path)
        return JsonResponse(data)
        


    def resize_image(input_path, output_path, target_size):
        # Open the input image
        image = Image.open(input_path)

        # Calculate the aspect ratios of the input and target sizes
        aspect_ratio_input = image.width / image.height
        aspect_ratio_target = target_size[0] / target_size[1]

        # Calculate the new dimensions based on the target size while preserving the aspect ratio
        if aspect_ratio_input > aspect_ratio_target:
            new_height = target_size[1]
            new_width = int(target_size[1] * aspect_ratio_input)
        else:
            new_width = target_size[0]
            new_height = int(target_size[0] / aspect_ratio_input)

        # Resize the image using the high-quality Lanczos interpolation method
        resized_image = image.resize((new_width, new_height), Image.BICUBIC)

        # Create a new blank image with the target size
        output_image = Image.new("RGB", target_size)

        # Calculate the position to paste the resized image, centering it on the canvas
        paste_position = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)

        # Paste the resized image onto the blank image
        output_image.paste(resized_image, paste_position)

        # Save the output image
        output_image.save(output_path)

    

    def get_video_fps(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Error opening video file.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps

        except Exception as e:
            print("Error:", e)
            return None