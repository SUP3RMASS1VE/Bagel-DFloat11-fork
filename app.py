import gradio as gr
import numpy as np
import os
import torch
import random
from tqdm import tqdm
import cv2
from typing import Tuple, Optional
import time

from accelerate import infer_auto_device_map, dispatch_model, init_empty_weights
from PIL import Image

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer

from dfloat11 import DFloat11Model


# Model Initialization
model_path = "./BAGEL-7B-MoT-DF11" # Download from https://huggingface.co/DFloat11/BAGEL-7B-MoT-DF11

print("🚀 Initializing BAGEL model...")
print(f"📁 Model path: {model_path}")

llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers -= 1

print("📦 Loading VAE model...")
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "vae/ae.safetensors"))

print("⚙️ Setting up model configuration...")
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

print("🏗️ Creating model architecture...")
with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

print("📝 Loading tokenizer...")
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

print("🔄 Setting up image transforms...")
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

print("💾 Loading model weights...")
model = model.to(torch.bfloat16)
model.load_state_dict({
    name: torch.empty(param.shape, dtype=param.dtype, device='cpu') if param.device.type == 'meta' else param
    for name, param in model.state_dict().items()
}, assign=True)

print("🔢 Applying DFloat11 quantization...")
DFloat11Model.from_pretrained(
    model_path,
    bfloat16_model=model,
    device='cpu',
)

print("🖥️ Setting up device mapping...")
# Model Loading and Multi GPU Infernece Preparing
device_map = infer_auto_device_map(
    model,
    max_memory={0: "24GiB"},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer", "SiglipVisionModel"],
)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
            
model = dispatch_model(model, device_map=device_map, force_hooks=True)
model = model.eval()

print("🔧 Initializing inferencer...")
# Inferencer Preparing 
inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

print("✅ Model initialization completed!")
print("🎉 Ready to generate images and understand content!")
print("-" * 50)

def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

# Text to Image function with thinking option and hyperparameters
def text_to_image(prompt, show_thinking=False, cfg_text_scale=4.0, cfg_interval=0.4, 
                 timestep_shift=3.0, num_timesteps=50, 
                 cfg_renorm_min=1.0, cfg_renorm_type="global", 
                 max_think_token_n=1024, do_sample=False, text_temperature=0.3,
                 seed=0, image_ratio="1:1", custom_width=1024, custom_height=1024, 
                 use_custom_resolution=False,
                 enable_upscaling=False, upscale_factor=2.0, upscale_method="lanczos"):
    # Set seed for reproducibility
    set_seed(seed)
    
    print(f"🎨 Starting text-to-image generation...")
    print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"⚙️ Settings: {num_timesteps} timesteps, CFG scale: {cfg_text_scale}")

    # Determine image resolution
    if use_custom_resolution:
        width, height = validate_resolution(custom_width, custom_height)
        image_shapes = (width, height)
        print(f"📐 Custom resolution: {width}x{height}")
    else:
        if image_ratio == "1:1":
            image_shapes = (1024, 1024)
        elif image_ratio == "4:3":
            image_shapes = (768, 1024)
        elif image_ratio == "3:4":
            image_shapes = (1024, 768) 
        elif image_ratio == "16:9":
            image_shapes = (576, 1024)
        elif image_ratio == "9:16":
            image_shapes = (1024, 576)
    
    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        image_shapes=image_shapes,
    )
    
    # Call inferencer with or without think parameter based on user choice
    result = inferencer(text=prompt, think=show_thinking, **inference_hyper)
    
    generated_image = result["image"]
    
    # Apply upscaling if enabled
    if enable_upscaling and upscale_factor > 1.0:
        print(f"⬆️ Upscaling image by {upscale_factor}x using {upscale_method}")
        try:
            if upscale_method in ["lanczos", "bicubic", "nearest"]:
                generated_image = simple_upscale(generated_image, upscale_factor, upscale_method)
            else:
                generated_image = opencv_upscale(generated_image, upscale_factor, upscale_method)
            print("✅ Upscaling completed!")
        except Exception as e:
            print(f"⚠️ Upscaling failed: {e}")
    
    print("✅ Image generation completed!")
    return generated_image, result.get("text", None)


# Image Understanding function with thinking option and hyperparameters
def image_understanding(image: Image.Image, prompt: str, show_thinking=False, 
                        do_sample=False, text_temperature=0.3, max_new_tokens=512):
    if image is None:
        return "Please upload an image."

    print(f"🔍 Starting image understanding...")
    print(f"❓ Question: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)
    
    # Set hyperparameters
    inference_hyper = dict(
        do_sample=do_sample,
        text_temperature=text_temperature,
        max_think_token_n=max_new_tokens, # Set max_length
    )
    
    # Use show_thinking parameter to control thinking process
    result = inferencer(image=image, text=prompt, think=show_thinking, 
                        understanding_output=True, **inference_hyper)
    
    print("✅ Image understanding completed!")
    return result["text"]


# Image Editing function with thinking option and hyperparameters
def edit_image(image: Image.Image, prompt: str, show_thinking=False, cfg_text_scale=4.0, 
              cfg_img_scale=2.0, cfg_interval=0.0, 
              timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=1.0, 
              cfg_renorm_type="text_channel", max_think_token_n=1024, 
              do_sample=False, text_temperature=0.3, seed=0,
              enable_upscaling=False, upscale_factor=2.0, upscale_method="lanczos"):
    # Set seed for reproducibility
    set_seed(seed)
    
    if image is None:
        return "Please upload an image.", ""

    print(f"✏️ Starting image editing...")
    print(f"📝 Edit instruction: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"⚙️ Settings: {num_timesteps} timesteps, Text CFG: {cfg_text_scale}, Image CFG: {cfg_img_scale}")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)
    
    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )
    
    # Include thinking parameter based on user choice
    result = inferencer(image=image, text=prompt, think=show_thinking, **inference_hyper)
    
    edited_image = result["image"]
    
    # Apply upscaling if enabled
    if enable_upscaling and upscale_factor > 1.0:
        print(f"⬆️ Upscaling image by {upscale_factor}x using {upscale_method}")
        try:
            if upscale_method in ["lanczos", "bicubic", "nearest"]:
                edited_image = simple_upscale(edited_image, upscale_factor, upscale_method)
            else:
                edited_image = opencv_upscale(edited_image, upscale_factor, upscale_method)
            print("✅ Upscaling completed!")
        except Exception as e:
            print(f"⚠️ Upscaling failed: {e}")
    
    print("✅ Image editing completed!")
    return edited_image, result.get("text", "")


# Helper function to load example images
def load_example_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading example image: {e}")
        return None


# Upscaling helper functions
def simple_upscale(image: Image.Image, scale_factor: float = 2.0, method: str = "lanczos") -> Image.Image:
    """Simple upscaling using PIL interpolation methods"""
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    if method == "lanczos":
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    elif method == "bicubic":
        return image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    elif method == "nearest":
        return image.resize((new_width, new_height), Image.Resampling.NEAREST)
    else:
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def opencv_upscale(image: Image.Image, scale_factor: float = 2.0, method: str = "edsr") -> Image.Image:
    """Advanced upscaling using OpenCV super-resolution methods"""
    try:
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if method == "edsr":
            # EDSR super-resolution (if available)
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            # Note: This requires downloading EDSR models
            # For now, fall back to bicubic
            width, height = image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            upscaled = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            # Bicubic interpolation
            width, height = image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            upscaled = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert back to PIL
        upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        return Image.fromarray(upscaled_rgb)
    except Exception as e:
        print(f"OpenCV upscaling failed: {e}, falling back to simple upscale")
        return simple_upscale(image, scale_factor, "bicubic")

def validate_resolution(width: int, height: int) -> Tuple[int, int]:
    """Validate and adjust resolution to reasonable limits"""
    # Ensure minimum size
    width = max(width, 256)
    height = max(height, 256)
    
    # Ensure maximum size (to prevent VRAM issues)
    max_pixels = 1024 * 1024 * 2  # 2MP max
    if width * height > max_pixels:
        ratio = np.sqrt(max_pixels / (width * height))
        width = int(width * ratio)
        height = int(height * ratio)
    
    # Ensure multiples of 64 for better model performance
    width = (width // 64) * 64
    height = (height // 64) * 64
    
    return width, height

# Conversational Chat functionality
class ConversationManager:
    def __init__(self, inferencer):
        self.inferencer = inferencer
        self.reset_conversation()
    
    def reset_conversation(self):
        self.conversation_history = []
        self.current_image = None
        self.gen_context = None
        self.initialized = False
    
    def add_to_history(self, role, content, image=None):
        self.conversation_history.append({
            'role': role,
            'content': content,
            'image': image,
            'timestamp': time.time()
        })
    
    def process_message(self, user_message, show_thinking=False, 
                       cfg_text_scale=4.0, cfg_img_scale=2.0, 
                       cfg_interval=0.4, timestep_shift=3.0, num_timesteps=50,
                       cfg_renorm_min=1.0, cfg_renorm_type="global",
                       max_think_token_n=1024, do_sample=False, 
                       text_temperature=0.3, seed=0, image_ratio="1:1",
                       custom_width=1024, custom_height=1024, use_custom_resolution=False,
                       enable_upscaling=False, upscale_factor=2.0, upscale_method="lanczos"):
        
        # Set seed for reproducibility
        set_seed(seed)
        
        # Add user message to history
        self.add_to_history("user", user_message)
        
        print(f"💬 Processing chat message: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
        
        # Determine if this is a request to generate or modify an image
        image_generation_keywords = [
            "generate", "create", "make", "draw", "paint", "show", "visualize", 
            "design", "produce", "render", "illustration", "picture", "image"
        ]
        
        image_modification_keywords = [
            "change", "modify", "edit", "alter", "adjust", "update", "fix", 
            "improve", "enhance", "different", "instead", "replace", "remove",
            "add", "make it", "make the", "but", "however"
        ]
        
        message_lower = user_message.lower()
        is_image_request = any(keyword in message_lower for keyword in image_generation_keywords)
        is_modification_request = (any(keyword in message_lower for keyword in image_modification_keywords) 
                                 and self.current_image is not None)
        
        try:
            if is_image_request or is_modification_request:
                # Image generation/modification request
                print("🎨 Detected image generation/modification request")
                
                # Determine image resolution
                if use_custom_resolution:
                    width, height = validate_resolution(custom_width, custom_height)
                    image_shapes = (width, height)
                    print(f"📐 Custom resolution: {width}x{height}")
                else:
                    if image_ratio == "1:1":
                        image_shapes = (1024, 1024)
                    elif image_ratio == "4:3":
                        image_shapes = (768, 1024)
                    elif image_ratio == "3:4":
                        image_shapes = (1024, 768) 
                    elif image_ratio == "16:9":
                        image_shapes = (576, 1024)
                    elif image_ratio == "9:16":
                        image_shapes = (1024, 576)
                
                # Prepare input list for inference
                input_list = []
                
                # Include current image if this is a modification request
                if is_modification_request and self.current_image is not None:
                    input_list.append(self.current_image)
                
                # Add the user's request
                input_list.append(user_message)
                
                # Set hyperparameters
                inference_hyper = dict(
                    max_think_token_n=max_think_token_n if show_thinking else 1024,
                    do_sample=do_sample if show_thinking else False,
                    text_temperature=text_temperature if show_thinking else 0.3,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=[cfg_interval, 1.0],
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    image_shapes=image_shapes,
                    think=show_thinking,
                )
                
                # Generate image
                result = self.inferencer.interleave_inference(input_list, **inference_hyper)
                
                # Extract image and text from result
                generated_image = None
                thinking_text = None
                
                for item in result:
                    if isinstance(item, Image.Image):
                        generated_image = item
                    elif isinstance(item, str):
                        thinking_text = item
                
                # Apply upscaling if enabled
                if enable_upscaling and upscale_factor > 1.0 and generated_image:
                    print(f"⬆️ Upscaling image by {upscale_factor}x using {upscale_method}")
                    try:
                        if upscale_method in ["lanczos", "bicubic", "nearest"]:
                            generated_image = simple_upscale(generated_image, upscale_factor, upscale_method)
                        else:
                            generated_image = opencv_upscale(generated_image, upscale_factor, upscale_method)
                        print("✅ Upscaling completed!")
                    except Exception as e:
                        print(f"⚠️ Upscaling failed: {e}")
                
                # Update current image and add to history
                if generated_image:
                    self.current_image = generated_image
                    response_text = "I've generated the image based on your request."
                    if thinking_text:
                        response_text = thinking_text + "\n\n" + response_text
                    
                    self.add_to_history("assistant", response_text, generated_image)
                    print("✅ Image generation completed!")
                    return self.format_chat_history(), generated_image, thinking_text if show_thinking else ""
                else:
                    error_msg = "Sorry, I couldn't generate an image from your request."
                    self.add_to_history("assistant", error_msg)
                    return self.format_chat_history(), None, ""
            
            else:
                # Text-only conversation
                print("💭 Processing text conversation")
                
                thinking_text = ""
                
                if show_thinking:
                    # Step 1: Generate thinking text only for understanding
                    print("💭 Starting thinking phase...")
                    try:
                        # Include current image in context if available
                        if self.current_image is not None:
                            thinking_result = self.inferencer(
                                image=self.current_image,
                                text=user_message, 
                                think=True, 
                                understanding_output=True,
                                do_sample=do_sample,
                                text_temperature=text_temperature,
                                max_think_token_n=max_think_token_n
                            )
                        else:
                            thinking_result = self.inferencer(
                                text=user_message, 
                                think=True, 
                                understanding_output=True,
                                do_sample=do_sample,
                                text_temperature=text_temperature,
                                max_think_token_n=max_think_token_n
                            )
                            thinking_text = thinking_result.get("text", "")
                            
                            if thinking_text:
                                print("💭 Thinking completed!")
                                # Show thinking immediately
                                current_history = self.format_chat_history()
                                yield current_history, self.current_image, thinking_text
                            
                    except Exception as e:
                        print(f"⚠️ Thinking failed: {e}")
                        thinking_text = "I apologize, but I encountered an error while thinking about your message."
                        current_history = self.format_chat_history()
                        yield current_history, self.current_image, thinking_text
                
                # Step 2: Generate final text response
                print("💬 Starting text response generation...")
                try:
                    if show_thinking and thinking_text:
                        # Include thinking text in the response context
                        full_message = f"{thinking_text}\n\n{user_message}"
                    else:
                        full_message = user_message
                    
                    # Set hyperparameters for understanding
                    inference_hyper = dict(
                        understanding_output=True,
                        think=False,  # Don't think again, we already did that
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        max_think_token_n=max_think_token_n,
                    )
                    
                    # Generate text response
                    if self.current_image is not None:
                        result = self.inferencer(
                            image=self.current_image,
                            text=full_message, 
                            **inference_hyper
                        )
                    else:
                        result = self.inferencer(
                            text=full_message, 
                            **inference_hyper
                        )
                    
                    response_text = result.get("text", "")
                    
                    if response_text:
                        self.add_to_history("assistant", response_text)
                        print("✅ Text response generated!")
                        final_history = self.format_chat_history()
                        yield final_history, self.current_image, thinking_text if show_thinking else ""
                    else:
                        error_msg = "Sorry, I couldn't process your message."
                        self.add_to_history("assistant", error_msg)
                        final_history = self.format_chat_history()
                        yield final_history, self.current_image, ""
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error while processing your message: {str(e)}"
                    print(f"❌ Text processing failed: {error_msg}")
                    self.add_to_history("assistant", error_msg)
                    final_history = self.format_chat_history()
                    yield final_history, self.current_image, ""
        
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(f"❌ Error in conversation: {error_msg}")
            self.add_to_history("assistant", error_msg) 
            return self.format_chat_history(), self.current_image, ""
    
    def format_chat_history(self):
        formatted_history = []
        for msg in self.conversation_history:
            role_emoji = "👤" if msg['role'] == 'user' else "🤖"
            content = f"{role_emoji} **{msg['role'].title()}**: {msg['content']}"
            if msg.get('image') is not None:
                content += "\n🖼️ *[Image generated]*"
            formatted_history.append(content)
        return "\n\n".join(formatted_history)

# Initialize conversation manager
conversation_manager = ConversationManager(inferencer)

# Gradio UI 
with gr.Blocks(title="Bagel-DFloat11 (SUP3R Edition)") as demo:
    gr.Markdown("""
<div style="display: flex; align-items: center; gap: 20px; padding: 20px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px; padding-left: 30px; padding-right: 30px;">
  <img src="https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/banner.png" alt="BAGEL" width="120" style="border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);"/>
  <div style="display: flex; align-items: center; gap: 15px;">
    <span style="font-size: 48px; line-height: 1;">🥯</span>
    <div>
      <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        Bagel-DFloat11 (SUP3R Edition)
      </h1>
      <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 1.1rem; font-weight: 300;">
        Advanced AI Model for Text-to-Image Generation & Understanding
      </p>
    </div>
  </div>
</div>
""")

    with gr.Tab("📝 Text to Image"):
        txt_input = gr.Textbox(
            label="Prompt", 
            value="A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere."
        )
        
        with gr.Row():
            show_thinking = gr.Checkbox(label="Thinking", value=False)
        
        # Add hyperparameter controls in an accordion
        with gr.Accordion("Inference Hyperparameters", open=False):
            # 参数一排两个布局
            with gr.Group():
                with gr.Row():
                    seed = gr.Slider(minimum=0, maximum=1000000, value=0, step=1, 
                                   label="Seed", info="0 for random seed, positive for reproducible results")
                    use_custom_resolution = gr.Checkbox(label="Custom Resolution", value=False, 
                                                      info="Use exact pixel dimensions instead of ratios")
                    
                with gr.Row():
                    # Resolution controls
                    image_ratio = gr.Dropdown(choices=["1:1", "4:3", "3:4", "16:9", "9:16"], 
                                                value="1:1", label="Image Ratio", 
                                                info="The longer size is fixed to 1024", visible=True)
                    custom_width = gr.Slider(minimum=256, maximum=2048, value=1024, step=64,
                                           label="Custom Width", info="Width in pixels (will be adjusted to multiples of 64)", visible=False)
                    custom_height = gr.Slider(minimum=256, maximum=2048, value=1024, step=64,
                                            label="Custom Height", info="Height in pixels (will be adjusted to multiples of 64)", visible=False)
                    
                with gr.Row():
                    cfg_text_scale = gr.Slider(minimum=1.0, maximum=8.0, value=4.0, step=0.1, interactive=True,
                                             label="CFG Text Scale", info="Controls how strongly the model follows the text prompt (4.0-8.0)")
                    cfg_interval = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.1, 
                                           label="CFG Interval", info="Start of CFG application interval (end is fixed at 1.0)")
                
                with gr.Row():
                    cfg_renorm_type = gr.Dropdown(choices=["global", "local", "text_channel"], 
                                                value="global", label="CFG Renorm Type", 
                                                info="If the genrated image is blurry, use 'global'")
                    cfg_renorm_min = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True,
                                             label="CFG Renorm Min", info="1.0 disables CFG-Renorm")
                
                with gr.Row():
                    num_timesteps = gr.Slider(minimum=10, maximum=100, value=50, step=5, interactive=True,
                                            label="Timesteps", info="Total denoising steps")
                    timestep_shift = gr.Slider(minimum=1.0, maximum=5.0, value=3.0, step=0.5, interactive=True,
                                             label="Timestep Shift", info="Higher values for layout, lower for details")
                
                # Upscaling controls
                with gr.Accordion("🔍 Post-Processing & Upscaling", open=False):
                    with gr.Row():
                        enable_upscaling = gr.Checkbox(label="Enable Upscaling", value=False, 
                                                     info="Apply upscaling after generation")
                        upscale_factor = gr.Slider(minimum=1.0, maximum=4.0, value=2.0, step=0.5,
                                                 label="Upscale Factor", info="How much to upscale the image")
                    with gr.Row():
                        upscale_method = gr.Dropdown(choices=["lanczos", "bicubic", "nearest", "opencv_bicubic"], 
                                                   value="lanczos", label="Upscale Method", 
                                                   info="Upscaling algorithm (lanczos is best for most cases)")
                
                # Thinking parameters in a single row
                thinking_params = gr.Group(visible=False)
                with thinking_params:
                    with gr.Row():
                        do_sample = gr.Checkbox(label="Sampling", value=False, info="Enable sampling for text generation")
                        max_think_token_n = gr.Slider(minimum=64, maximum=4006, value=1024, step=64, interactive=True,
                                                    label="Max Think Tokens", info="Maximum number of tokens for thinking")
                        text_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, interactive=True,
                                                  label="Temperature", info="Controls randomness in text generation")
        
        thinking_output = gr.Textbox(label="Thinking Process", visible=False)
        img_output = gr.Image(label="Generated Image")
        gen_btn = gr.Button("Generate")
        
        # Dynamically show/hide thinking process box and parameters
        def update_thinking_visibility(show):
            return gr.update(visible=show), gr.update(visible=show)
        
        # Dynamically show/hide resolution controls
        def update_resolution_visibility(use_custom):
            return (gr.update(visible=not use_custom), 
                    gr.update(visible=use_custom), 
                    gr.update(visible=use_custom))
        
        show_thinking.change(
            fn=update_thinking_visibility,
            inputs=[show_thinking],
            outputs=[thinking_output, thinking_params]
        )
        
        use_custom_resolution.change(
            fn=update_resolution_visibility,
            inputs=[use_custom_resolution],
            outputs=[image_ratio, custom_width, custom_height]
        )
        
        # Process function based on thinking option and hyperparameters
        def process_text_to_image(prompt, show_thinking, cfg_text_scale, 
                                 cfg_interval, timestep_shift, 
                                 num_timesteps, cfg_renorm_min, cfg_renorm_type, 
                                 max_think_token_n, do_sample, text_temperature, seed, 
                                 image_ratio, custom_width, custom_height, use_custom_resolution,
                                 enable_upscaling, upscale_factor, upscale_method):
            
            # Set seed for reproducibility
            set_seed(seed)
            
            print(f"🎨 Starting text-to-image generation...")
            print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"⚙️ Settings: {num_timesteps} timesteps, CFG scale: {cfg_text_scale}")

            # Determine image resolution
            if use_custom_resolution:
                width, height = validate_resolution(custom_width, custom_height)
                image_shapes = (width, height)
                print(f"📐 Custom resolution: {width}x{height}")
            else:
                if image_ratio == "1:1":
                    image_shapes = (1024, 1024)
                elif image_ratio == "4:3":
                    image_shapes = (768, 1024)
                elif image_ratio == "3:4":
                    image_shapes = (1024, 768) 
                elif image_ratio == "16:9":
                    image_shapes = (576, 1024)
                elif image_ratio == "9:16":
                    image_shapes = (1024, 576)
            
            thinking_text = ""
            
            if show_thinking:
                # Step 1: Generate thinking text only
                print("💭 Starting thinking phase...")
                try:
                    # Create a thinking-only inference
                    thinking_result = inferencer(
                        text=prompt, 
                        think=True, 
                        understanding_output=True,
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        max_think_token_n=max_think_token_n
                    )
                    thinking_text = thinking_result.get("text", "")
                    
                    if thinking_text:
                        print("💭 Thinking completed!")
                        # Step 2: Show thinking text immediately (yield for real-time display)
                        yield None, thinking_text
                    
                except Exception as e:
                    print(f"⚠️ Thinking failed: {e}")
                    thinking_text = "I apologize, but I encountered an error while thinking about your request."
                    yield None, thinking_text
            
            # Step 3: Generate image
            print("🖼️ Starting image generation...")
            try:
                if show_thinking and thinking_text:
                    # Include thinking text in the prompt for image generation
                    full_prompt = f"{thinking_text}\n\n{prompt}"
                else:
                    full_prompt = prompt
                
                # Set hyperparameters for image generation (no cfg_img_scale for text-to-image)
                inference_hyper = dict(
                    think=False,  # Don't think again, we already did that
                    cfg_text_scale=cfg_text_scale,
                    cfg_interval=[cfg_interval, 1.0],
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    image_shapes=image_shapes,
                )
                
                # Generate image
                result = inferencer(text=full_prompt, **inference_hyper)
                generated_image = result["image"]
                
                # Apply upscaling if enabled
                if enable_upscaling and upscale_factor > 1.0 and generated_image:
                    if show_thinking:
                        yield None, thinking_text + f"\n\n⬆️ Upscaling image by {upscale_factor}x using {upscale_method}..."
                    print(f"⬆️ Upscaling image by {upscale_factor}x using {upscale_method}")
                    try:
                        if upscale_method in ["lanczos", "bicubic", "nearest"]:
                            generated_image = simple_upscale(generated_image, upscale_factor, upscale_method)
                        else:
                            generated_image = opencv_upscale(generated_image, upscale_factor, upscale_method)
                        print("✅ Upscaling completed!")
                    except Exception as e:
                        print(f"⚠️ Upscaling failed: {e}")
                
                print("✅ Image generation completed!")
                # Final yield with exactly 2 values: image and thinking text
                yield generated_image, thinking_text if show_thinking else ""
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error while generating the image: {str(e)}"
                print(f"❌ Image generation failed: {error_msg}")
                # Return error with exactly 2 values
                if show_thinking:
                    yield None, thinking_text + f"\n\n{error_msg}" if thinking_text else error_msg
                else:
                    yield None, error_msg
        
        gen_btn.click(
            fn=process_text_to_image,
            inputs=[
                txt_input, show_thinking, cfg_text_scale, 
                cfg_interval, timestep_shift, 
                num_timesteps, cfg_renorm_min, cfg_renorm_type,
                max_think_token_n, do_sample, text_temperature, seed, 
                image_ratio, custom_width, custom_height, use_custom_resolution,
                enable_upscaling, upscale_factor, upscale_method
            ],
            outputs=[img_output, thinking_output]
        )

    with gr.Tab("🖌️ Image Edit"):
        with gr.Row():
            with gr.Column(scale=1):
                edit_image_input = gr.Image(label="Input Image", value=load_example_image('test_images/women.jpg'))
                edit_prompt = gr.Textbox(
                    label="Prompt",
                    value="She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes."
                )
            
            with gr.Column(scale=1):
                edit_image_output = gr.Image(label="Result")
                edit_thinking_output = gr.Textbox(label="Thinking Process", visible=False)
        
        with gr.Row():
            edit_show_thinking = gr.Checkbox(label="Thinking", value=False)
        
        # Add hyperparameter controls in an accordion
        with gr.Accordion("Inference Hyperparameters", open=False):
            with gr.Group():
                with gr.Row():
                    edit_seed = gr.Slider(minimum=0, maximum=1000000, value=0, step=1, interactive=True,
                                        label="Seed", info="0 for random seed, positive for reproducible results")
                    edit_cfg_text_scale = gr.Slider(minimum=1.0, maximum=8.0, value=4.0, step=0.1, interactive=True,
                                                  label="CFG Text Scale", info="Controls how strongly the model follows the text prompt")
                
                with gr.Row():
                    edit_cfg_img_scale = gr.Slider(minimum=1.0, maximum=4.0, value=2.0, step=0.1, interactive=True,
                                                 label="CFG Image Scale", info="Controls how much the model preserves input image details")
                    edit_cfg_interval = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True,
                                                label="CFG Interval", info="Start of CFG application interval (end is fixed at 1.0)")
                    
                with gr.Row():
                    edit_cfg_renorm_type = gr.Dropdown(choices=["global", "local", "text_channel"], 
                                                     value="text_channel", label="CFG Renorm Type", 
                                                     info="If the genrated image is blurry, use 'global")
                    edit_cfg_renorm_min = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True,
                                                  label="CFG Renorm Min", info="1.0 disables CFG-Renorm")
                
                with gr.Row():
                    edit_num_timesteps = gr.Slider(minimum=10, maximum=100, value=50, step=5, interactive=True,
                                                 label="Timesteps", info="Total denoising steps")
                    edit_timestep_shift = gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.5, interactive=True,
                                                  label="Timestep Shift", info="Higher values for layout, lower for details")
                
                # Upscaling controls for editing
                with gr.Accordion("🔍 Post-Processing & Upscaling", open=False):
                    with gr.Row():
                        edit_enable_upscaling = gr.Checkbox(label="Enable Upscaling", value=False, 
                                                          info="Apply upscaling after editing")
                        edit_upscale_factor = gr.Slider(minimum=1.0, maximum=4.0, value=2.0, step=0.5,
                                                      label="Upscale Factor", info="How much to upscale the image")
                    with gr.Row():
                        edit_upscale_method = gr.Dropdown(choices=["lanczos", "bicubic", "nearest", "opencv_bicubic"], 
                                                        value="lanczos", label="Upscale Method", 
                                                        info="Upscaling algorithm (lanczos is best for most cases)")
                
                # Thinking parameters in a single row
                edit_thinking_params = gr.Group(visible=False)
                with edit_thinking_params:
                    with gr.Row():
                        edit_do_sample = gr.Checkbox(label="Sampling", value=False, info="Enable sampling for text generation")
                        edit_max_think_token_n = gr.Slider(minimum=64, maximum=4006, value=1024, step=64, interactive=True,
                                                         label="Max Think Tokens", info="Maximum number of tokens for thinking")
                        edit_text_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, interactive=True,
                                                        label="Temperature", info="Controls randomness in text generation")
        
        edit_btn = gr.Button("Submit")
        
        # Dynamically show/hide thinking process box for editing
        def update_edit_thinking_visibility(show):
            return gr.update(visible=show), gr.update(visible=show)
        
        edit_show_thinking.change(
            fn=update_edit_thinking_visibility,
            inputs=[edit_show_thinking],
            outputs=[edit_thinking_output, edit_thinking_params]
        )
        
        # Process editing with thinking option and hyperparameters
        def process_edit_image(image, prompt, show_thinking, cfg_text_scale, 
                              cfg_img_scale, cfg_interval, 
                              timestep_shift, num_timesteps, cfg_renorm_min, 
                              cfg_renorm_type, max_think_token_n, do_sample, 
                              text_temperature, seed, enable_upscaling, 
                              upscale_factor, upscale_method):
            
            # Set seed for reproducibility
            set_seed(seed)
            
            if image is None:
                yield "Please upload an image.", ""
                return

            print(f"✏️ Starting image editing...")
            print(f"📝 Edit instruction: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"⚙️ Settings: {num_timesteps} timesteps, Text CFG: {cfg_text_scale}, Image CFG: {cfg_img_scale}")

            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            image = pil_img2rgb(image)
            
            thinking_text = ""
            
            if show_thinking:
                # Step 1: Generate thinking text only
                print("💭 Starting thinking phase...")
                try:
                    # Create a thinking-only inference with image context
                    thinking_result = inferencer(
                        image=image,
                        text=prompt, 
                        think=True, 
                        understanding_output=True,
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        max_think_token_n=max_think_token_n
                    )
                    thinking_text = thinking_result.get("text", "")
                    
                    if thinking_text:
                        print("💭 Thinking completed!")
                        # Step 2: Show thinking text immediately
                        yield None, thinking_text
                    
                except Exception as e:
                    print(f"⚠️ Thinking failed: {e}")
                    thinking_text = "I apologize, but I encountered an error while thinking about your editing request."
                    yield None, thinking_text
            
            # Step 3: Generate edited image
            print("🖼️ Starting image editing...")
            try:
                if show_thinking and thinking_text:
                    # Include thinking text in the prompt for image editing
                    full_prompt = f"{thinking_text}\n\n{prompt}"
                else:
                    full_prompt = prompt
                
                # Set hyperparameters for image editing
                inference_hyper = dict(
                    think=False,  # Don't think again, we already did that
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=[cfg_interval, 1.0],
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )
                
                # Generate edited image
                result = inferencer(image=image, text=full_prompt, **inference_hyper)
                edited_image = result["image"]
                
                # Apply upscaling if enabled
                if enable_upscaling and upscale_factor > 1.0:
                    if show_thinking:
                        yield None, thinking_text + f"\n\n⬆️ Upscaling image by {upscale_factor}x using {upscale_method}..."
                    print(f"⬆️ Upscaling image by {upscale_factor}x using {upscale_method}")
                    try:
                        if upscale_method in ["lanczos", "bicubic", "nearest"]:
                            edited_image = simple_upscale(edited_image, upscale_factor, upscale_method)
                        else:
                            edited_image = opencv_upscale(edited_image, upscale_factor, upscale_method)
                        print("✅ Upscaling completed!")
                    except Exception as e:
                        print(f"⚠️ Upscaling failed: {e}")
                
                print("✅ Image editing completed!")
                # Final yield with exactly 2 values: image and thinking text
                yield edited_image, thinking_text if show_thinking else ""
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error while editing the image: {str(e)}"
                print(f"❌ Image editing failed: {error_msg}")
                # Return error with exactly 2 values
                if show_thinking:
                    yield None, thinking_text + f"\n\n{error_msg}" if thinking_text else error_msg
                else:
                    yield None, error_msg
        
        edit_btn.click(
            fn=process_edit_image,
            inputs=[
                edit_image_input, edit_prompt, edit_show_thinking, 
                edit_cfg_text_scale, edit_cfg_img_scale, edit_cfg_interval,
                edit_timestep_shift, edit_num_timesteps, 
                edit_cfg_renorm_min, edit_cfg_renorm_type,
                edit_max_think_token_n, edit_do_sample, edit_text_temperature, edit_seed,
                edit_enable_upscaling, edit_upscale_factor, edit_upscale_method
            ],
            outputs=[edit_image_output, edit_thinking_output]
        )

    with gr.Tab("🖼️ Image Understanding"):
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(label="Input Image", value=load_example_image('test_images/meme.jpg'))
                understand_prompt = gr.Textbox(
                    label="Prompt", 
                    value="Can someone explain what's funny about this meme??"
                )
            
            with gr.Column(scale=1):
                txt_output = gr.Textbox(label="Result", lines=20)
        
        with gr.Row():
            understand_show_thinking = gr.Checkbox(label="Thinking", value=False)
        
        # Add hyperparameter controls in an accordion
        with gr.Accordion("Inference Hyperparameters", open=False):
            with gr.Row():
                understand_do_sample = gr.Checkbox(label="Sampling", value=False, info="Enable sampling for text generation")
                understand_text_temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, interactive=True,
                                                     label="Temperature", info="Controls randomness in text generation (0=deterministic, 1=creative)")
                understand_max_new_tokens = gr.Slider(minimum=64, maximum=4096, value=512, step=64, interactive=True,
                                                   label="Max New Tokens", info="Maximum length of generated text, including potential thinking")
        
        img_understand_btn = gr.Button("Submit")
        
        # Process understanding with thinking option and hyperparameters
        def process_understanding(image, prompt, show_thinking, do_sample, 
                                 text_temperature, max_new_tokens):
            result = image_understanding(
                image, prompt, show_thinking, do_sample, 
                text_temperature, max_new_tokens
            )
            return result
        
        img_understand_btn.click(
            fn=process_understanding,
            inputs=[
                img_input, understand_prompt, understand_show_thinking,
                understand_do_sample, understand_text_temperature, understand_max_new_tokens
            ],
            outputs=txt_output
        )

    with gr.Tab("💬 Chat with BAGEL"):
        gr.Markdown("""
        ### 🗣️ Conversational AI with Image Generation & Understanding
        
        **How to use:**
        - Ask me to generate images: *"Create a sunset over mountains"*
        - Ask about the current image: *"What colors are in this image?"*
        - Request modifications: *"Make it more colorful"* or *"Change the sky to night"*
        - Have normal conversations while maintaining context
        
        The model remembers our conversation and the current image context!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                chat_display = gr.Textbox(
                    label="Conversation",
                    value="",
                    lines=15,
                    max_lines=25,
                    interactive=False,
                    show_copy_button=True
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Your message",
                        placeholder="Ask me anything or request an image...",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send 💬", scale=1, variant="primary")
                    clear_btn = gr.Button("Clear Chat 🗑️", scale=1)
            
            with gr.Column(scale=1):
                chat_image_output = gr.Image(label="Current Image", height=400)
                chat_thinking_output = gr.Textbox(label="AI Thinking Process", lines=8, visible=False)
        
        # Chat controls
        with gr.Row():
            chat_show_thinking = gr.Checkbox(label="Show AI Thinking", value=False)
        
        # Chat hyperparameters
        with gr.Accordion("🔧 Chat Settings", open=False):
            with gr.Row():
                chat_seed = gr.Slider(minimum=0, maximum=1000000, value=0, step=1,
                                    label="Seed", info="0 for random seed")
                chat_use_custom_resolution = gr.Checkbox(label="Custom Resolution", value=False, 
                                                        info="Use exact pixel dimensions instead of ratios")
            
            with gr.Row():
                # Resolution controls
                chat_image_ratio = gr.Dropdown(choices=["1:1", "4:3", "3:4", "16:9", "9:16"], 
                                              value="1:1", label="Image Ratio", 
                                              info="The longer size is fixed to 1024", visible=True)
                chat_custom_width = gr.Slider(minimum=256, maximum=2048, value=1024, step=64,
                                            label="Custom height", info="height in pixels (will be adjusted to multiples of 64)", visible=False)
                chat_custom_height = gr.Slider(minimum=256, maximum=2048, value=1024, step=64,
                                             label="Custom Width", info="Width in pixels (will be adjusted to multiples of 64)", visible=False)
            
            with gr.Row():
                chat_cfg_text_scale = gr.Slider(minimum=1.0, maximum=8.0, value=4.0, step=0.1,
                                              label="CFG Text Scale")
                chat_cfg_img_scale = gr.Slider(minimum=1.0, maximum=4.0, value=2.0, step=0.1,
                                             label="CFG Image Scale")
            
            with gr.Row():
                chat_cfg_interval = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.1,
                                            label="CFG Interval", info="Start of CFG application interval (end is fixed at 1.0)")
                chat_timestep_shift = gr.Slider(minimum=1.0, maximum=5.0, value=3.0, step=0.5,
                                              label="Timestep Shift", info="Higher values for layout, lower for details")
            
            with gr.Row():
                chat_cfg_renorm_min = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                                              label="CFG Renorm Min", info="1.0 disables CFG-Renorm (keep at 0.0 for sharp images)")
                chat_cfg_renorm_type = gr.Dropdown(choices=["global", "local", "text_channel"], 
                                                 value="global", label="CFG Renorm Type", 
                                                 info="If the generated image is blurry, use 'global'")
            
            with gr.Row():
                chat_num_timesteps = gr.Slider(minimum=10, maximum=100, value=50, step=5,
                                             label="Timesteps")
                chat_text_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1,
                                                label="Text Temperature")
            
            # Chat upscaling controls
            with gr.Accordion("🔍 Image Enhancement", open=False):
                with gr.Row():
                    chat_enable_upscaling = gr.Checkbox(label="Enable Upscaling", value=False)
                    chat_upscale_factor = gr.Slider(minimum=1.0, maximum=4.0, value=2.0, step=0.5,
                                                  label="Upscale Factor")
                    chat_upscale_method = gr.Dropdown(choices=["lanczos", "bicubic", "nearest", "opencv_bicubic"], 
                                                    value="lanczos", label="Upscale Method")
        
        # Function to handle chat messages
        def handle_chat_message(message, show_thinking, cfg_text_scale, cfg_img_scale, 
                               cfg_interval, timestep_shift, num_timesteps, cfg_renorm_min, 
                               cfg_renorm_type, max_think_token_n, do_sample, text_temperature, 
                               seed, image_ratio, custom_width, custom_height, use_custom_resolution,
                               enable_upscaling, upscale_factor, upscale_method):
            if not message.strip():
                yield "", None, ""
                return
            
            # Set seed for reproducibility
            set_seed(seed)
            
            # Add user message to history
            conversation_manager.add_to_history("user", message)
            
            print(f"💬 Processing chat message: {message[:100]}{'...' if len(message) > 100 else ''}")
            
            # Show thinking progress if enabled
            if show_thinking:
                current_history = conversation_manager.format_chat_history()
                yield current_history, conversation_manager.current_image, "🤔 Thinking about your message..."
            
            # Determine if this is a request to generate or modify an image
            image_generation_keywords = [
                "generate", "create", "make", "draw", "paint", "show", "visualize", 
                "design", "produce", "render", "illustration", "picture", "image"
            ]
            
            image_modification_keywords = [
                "change", "modify", "edit", "alter", "adjust", "update", "fix", 
                "improve", "enhance", "different", "instead", "replace", "remove",
                "add", "make it", "make the", "but", "however"
            ]
            
            message_lower = message.lower()
            is_image_request = any(keyword in message_lower for keyword in image_generation_keywords)
            is_modification_request = (any(keyword in message_lower for keyword in image_modification_keywords) 
                                     and conversation_manager.current_image is not None)
            
            try:
                if is_image_request or is_modification_request:
                    # Image generation/modification request
                    print("🎨 Detected image generation/modification request")
                    
                    # Determine image resolution
                    if use_custom_resolution:
                        width, height = validate_resolution(custom_width, custom_height)
                        image_shapes = (width, height)
                        print(f"📐 Custom resolution: {width}x{height}")
                    else:
                        if image_ratio == "1:1":
                            image_shapes = (1024, 1024)
                        elif image_ratio == "4:3":
                            image_shapes = (768, 1024)
                        elif image_ratio == "3:4":
                            image_shapes = (1024, 768) 
                        elif image_ratio == "16:9":
                            image_shapes = (576, 1024)
                        elif image_ratio == "9:16":
                            image_shapes = (1024, 576)
                    
                    # Prepare input list for inference
                    input_list = []
                    
                    # Include current image if this is a modification request
                    if is_modification_request and conversation_manager.current_image is not None:
                        input_list.append(conversation_manager.current_image)
                    
                    # Add the user's request
                    input_list.append(message)
                    
                    thinking_text = ""
                    
                    if show_thinking:
                        # Step 1: Generate thinking text only
                        print("💭 Starting thinking phase...")
                        try:
                            # Create a thinking-only inference
                            if is_modification_request and conversation_manager.current_image is not None:
                                thinking_result = conversation_manager.inferencer(
                                    image=conversation_manager.current_image,
                                    text=message, 
                                    think=True, 
                                    understanding_output=True,
                                    do_sample=do_sample,
                                    text_temperature=text_temperature,
                                    max_think_token_n=max_think_token_n
                                )
                            else:
                                thinking_result = conversation_manager.inferencer(
                                    text=message, 
                                    think=True, 
                                    understanding_output=True,
                                    do_sample=do_sample,
                                    text_temperature=text_temperature,
                                    max_think_token_n=max_think_token_n
                                )
                            thinking_text = thinking_result.get("text", "")
                            
                            if thinking_text:
                                print("💭 Thinking completed!")
                                # Show thinking immediately
                                current_history = conversation_manager.format_chat_history()
                                yield current_history, conversation_manager.current_image, thinking_text
                            
                        except Exception as e:
                            print(f"⚠️ Thinking failed: {e}")
                            thinking_text = "I apologize, but I encountered an error while thinking about your request."
                            current_history = conversation_manager.format_chat_history()
                            yield current_history, conversation_manager.current_image, thinking_text
                    
                    # Step 2: Generate image
                    print("🖼️ Starting image generation...")
                    try:
                        if show_thinking and thinking_text:
                            # Include thinking text in the prompt for image generation
                            full_message = f"{thinking_text}\n\n{message}"
                        else:
                            full_message = message
                        
                        # Set hyperparameters
                        inference_hyper = dict(
                            think=False,  # Don't think again, we already did that
                            cfg_text_scale=cfg_text_scale,
                            cfg_img_scale=cfg_img_scale,
                            cfg_interval=[cfg_interval, 1.0],
                            timestep_shift=timestep_shift,
                            num_timesteps=num_timesteps,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                            image_shapes=image_shapes,
                        )
                        
                        # Generate image
                        if is_modification_request and conversation_manager.current_image is not None:
                            result = conversation_manager.inferencer(
                                image=conversation_manager.current_image, 
                                text=full_message, 
                                **inference_hyper
                            )
                        else:
                            result = conversation_manager.inferencer(
                                text=full_message, 
                                **inference_hyper
                            )
                        
                        generated_image = result.get("image")
                        
                        # Apply upscaling if enabled
                        if enable_upscaling and upscale_factor > 1.0 and generated_image:
                            if show_thinking:
                                current_history = conversation_manager.format_chat_history()
                                upscale_message = thinking_text + f"\n\n⬆️ Upscaling image by {upscale_factor}x using {upscale_method}..."
                                yield current_history, conversation_manager.current_image, upscale_message
                            print(f"⬆️ Upscaling image by {upscale_factor}x using {upscale_method}")
                            try:
                                if upscale_method in ["lanczos", "bicubic", "nearest"]:
                                    generated_image = simple_upscale(generated_image, upscale_factor, upscale_method)
                                else:
                                    generated_image = opencv_upscale(generated_image, upscale_factor, upscale_method)
                                print("✅ Upscaling completed!")
                            except Exception as e:
                                print(f"⚠️ Upscaling failed: {e}")
                        
                        # Update current image and add to history
                        if generated_image:
                            conversation_manager.current_image = generated_image
                            response_text = "I've generated the image based on your request."
                            if thinking_text:
                                response_text = thinking_text + "\n\n" + response_text
                            
                            conversation_manager.add_to_history("assistant", response_text, generated_image)
                            print("✅ Image generation completed!")
                            final_history = conversation_manager.format_chat_history()
                            yield final_history, generated_image, thinking_text if show_thinking else ""
                        else:
                            error_msg = "Sorry, I couldn't generate an image from your request."
                            conversation_manager.add_to_history("assistant", error_msg)
                            final_history = conversation_manager.format_chat_history()
                            yield final_history, None, ""
                            
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error while generating the image: {str(e)}"
                        print(f"❌ Image generation failed: {error_msg}")
                        conversation_manager.add_to_history("assistant", error_msg) 
                        final_history = conversation_manager.format_chat_history()
                        yield final_history, conversation_manager.current_image, ""
                
                else:
                    # Text-only conversation
                    print("💭 Processing text conversation")
                    
                    thinking_text = ""
                    
                    if show_thinking:
                        # Step 1: Generate thinking text only for understanding
                        print("💭 Starting thinking phase...")
                        try:
                            # Include current image in context if available
                            if conversation_manager.current_image is not None:
                                thinking_result = conversation_manager.inferencer(
                                    image=conversation_manager.current_image,
                                    text=message, 
                                    think=True, 
                                    understanding_output=True,
                                    do_sample=do_sample,
                                    text_temperature=text_temperature,
                                    max_think_token_n=max_think_token_n
                                )
                            else:
                                thinking_result = conversation_manager.inferencer(
                                    text=message, 
                                    think=True, 
                                    understanding_output=True,
                                    do_sample=do_sample,
                                    text_temperature=text_temperature,
                                    max_think_token_n=max_think_token_n
                                )
                                thinking_text = thinking_result.get("text", "")
                                
                                if thinking_text:
                                    print("💭 Thinking completed!")
                                    # Show thinking immediately
                                    current_history = conversation_manager.format_chat_history()
                                    yield current_history, conversation_manager.current_image, thinking_text
                                
                        except Exception as e:
                            print(f"⚠️ Thinking failed: {e}")
                            thinking_text = "I apologize, but I encountered an error while thinking about your message."
                            current_history = conversation_manager.format_chat_history()
                            yield current_history, conversation_manager.current_image, thinking_text
                        
                        # Step 2: Generate final text response
                        print("💬 Starting text response generation...")
                        try:
                            if show_thinking and thinking_text:
                                # Include thinking text in the response context
                                full_message = f"{thinking_text}\n\n{message}"
                            else:
                                full_message = message
                            
                            # Set hyperparameters for understanding
                            inference_hyper = dict(
                                understanding_output=True,
                                think=False,  # Don't think again, we already did that
                                do_sample=do_sample,
                                text_temperature=text_temperature,
                                max_think_token_n=max_think_token_n,
                            )
                            
                            # Generate text response
                            if conversation_manager.current_image is not None:
                                result = conversation_manager.inferencer(
                                    image=conversation_manager.current_image,
                                    text=full_message, 
                                    **inference_hyper
                                )
                            else:
                                result = conversation_manager.inferencer(
                                    text=full_message, 
                                    **inference_hyper
                                )
                            
                            response_text = result.get("text", "")
                            
                            if response_text:
                                conversation_manager.add_to_history("assistant", response_text)
                                print("✅ Text response generated!")
                                final_history = conversation_manager.format_chat_history()
                                yield final_history, conversation_manager.current_image, thinking_text if show_thinking else ""
                            else:
                                error_msg = "Sorry, I couldn't process your message."
                                conversation_manager.add_to_history("assistant", error_msg)
                                final_history = conversation_manager.format_chat_history()
                                yield final_history, conversation_manager.current_image, ""
                            
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error while processing your message: {str(e)}"
                            print(f"❌ Text processing failed: {error_msg}")
                            conversation_manager.add_to_history("assistant", error_msg)
                            final_history = conversation_manager.format_chat_history()
                            yield final_history, conversation_manager.current_image, ""
            
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                print(f"❌ Error in conversation: {error_msg}")
                conversation_manager.add_to_history("assistant", error_msg) 
                final_history = conversation_manager.format_chat_history()
                yield final_history, conversation_manager.current_image, ""
        
        # Function to clear chat
        def clear_chat():
            conversation_manager.reset_conversation()
            return "", None, ""
            
        # Function to update thinking visibility
        def update_chat_thinking_visibility(show):
            return gr.update(visible=show)
        
        # Dynamically show/hide resolution controls for chat
        def update_chat_resolution_visibility(use_custom):
            return (gr.update(visible=not use_custom), 
                    gr.update(visible=use_custom), 
                    gr.update(visible=use_custom))
        
        # Event handlers
        chat_show_thinking.change(
            fn=update_chat_thinking_visibility,
            inputs=[chat_show_thinking],
            outputs=[chat_thinking_output]
        )
        
        chat_use_custom_resolution.change(
            fn=update_chat_resolution_visibility,
            inputs=[chat_use_custom_resolution],
            outputs=[chat_image_ratio, chat_custom_width, chat_custom_height]
        )
        
        send_btn.click(
            fn=handle_chat_message,
            inputs=[
                chat_input, chat_show_thinking, chat_cfg_text_scale, chat_cfg_img_scale,
                chat_cfg_interval, chat_timestep_shift,
                chat_num_timesteps, chat_cfg_renorm_min, chat_cfg_renorm_type,
                gr.State(1024),  # max_think_token_n
                gr.State(False),  # do_sample
                chat_text_temperature, chat_seed, chat_image_ratio,
                chat_custom_width, chat_custom_height, chat_use_custom_resolution,
                chat_enable_upscaling, chat_upscale_factor, chat_upscale_method
            ],
            outputs=[chat_display, chat_image_output, chat_thinking_output]
        ).then(
            fn=lambda: "",  # Clear input after sending
            inputs=[],
            outputs=[chat_input]
        )
        
        chat_input.submit(
            fn=handle_chat_message,
            inputs=[
                chat_input, chat_show_thinking, chat_cfg_text_scale, chat_cfg_img_scale,
                chat_cfg_interval, chat_timestep_shift,
                chat_num_timesteps, chat_cfg_renorm_min, chat_cfg_renorm_type,
                gr.State(1024),  # max_think_token_n
                gr.State(False),  # do_sample
                chat_text_temperature, chat_seed, chat_image_ratio,
                chat_custom_width, chat_custom_height, chat_use_custom_resolution,
                chat_enable_upscaling, chat_upscale_factor, chat_upscale_method
            ],
            outputs=[chat_display, chat_image_output, chat_thinking_output]
        ).then(
            fn=lambda: "",  # Clear input after sending
            inputs=[],
            outputs=[chat_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chat_display, chat_image_output, chat_thinking_output]
        )

    gr.Markdown("""
<div style="display: flex; justify-content: flex-start; flex-wrap: wrap; gap: 10px;">
  <a href="https://bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Website-0A66C2?logo=safari&logoColor=white"
      alt="BAGEL Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2505.14683">
    <img
      src="https://img.shields.io/badge/BAGEL-Paper-red?logo=arxiv&logoColor=red"
      alt="BAGEL Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT">
    <img 
        src="https://img.shields.io/badge/BAGEL-Hugging%20Face-orange?logo=huggingface&logoColor=yellow" 
        alt="BAGEL on Hugging Face"
    />
  </a>
  <a href="https://demo.bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Demo-blue?logo=googleplay&logoColor=blue"
      alt="BAGEL Demo"
    />
  </a>
  <a href="https://discord.gg/Z836xxzy">
    <img
      src="https://img.shields.io/badge/BAGEL-Discord-5865F2?logo=discord&logoColor=purple"
      alt="BAGEL Discord"
    />
  </a>
  <a href="mailto:bagel@bytedance.com">
    <img
      src="https://img.shields.io/badge/BAGEL-Email-D14836?logo=gmail&logoColor=red"
      alt="BAGEL Email"
    />
  </a>
</div>
""")

demo.launch(share=False)
