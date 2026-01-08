from flask import Flask, request, send_file, jsonify, render_template
import torch
from diffusers import DiffusionPipeline
from io import BytesIO
import logging
import random
import gc
import time

# ──────────────────────────────────────────────────────
# 🔧 SETUP
# ──────────────────────────────────────────────────────

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def cleanup_memory():
    """Clean up GPU memory to prevent out-of-memory errors"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Model configuration
base_model = "Lykon/dreamshaper-xl-v2-turbo"
lora_path = "lora_weights/genestate_lora_final"

logging.info("🔄 Lade Dreamshaper XL Turbo...")

try:
    cleanup_memory()
    
    # Load base model
    pipe = DiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    
    # Load LoRA weights
    try:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=1.0)
        logging.info("✅ LoRA geladen und gefused")
    except Exception as e:
        logging.warning(f"⚠️  LoRA Fehler: {str(e)}")
    
    pipe.to("cuda")
    
    # Enable optimizations
    pipe.enable_vae_tiling()
    
    # Warm-up generation
    logging.info("🔥 Warming up...")
    with torch.inference_mode():
        _ = pipe(
            prompt="4k, medium modern villa",
            num_inference_steps=12,
            guidance_scale=3.5,
            width=864,
            height=864,
        ).images[0]
    cleanup_memory()
    
    logging.info("✅ Ready!")
    
except Exception as e:
    logging.error(f"❌ Fehler: {str(e)}")
    raise

# ──────────────────────────────────────────────────────
# 🖼️ IMAGE GENERATION
# ──────────────────────────────────────────────────────

def generate_house_image(style, size, roof_color, extra):
    """
    Generate house image based on user parameters
    
    Args:
        style: 'modern', 'wooden', or 'mediterranean'
        size: 'small', 'medium', or 'large'
        roof_color: 'dark', 'red', or 'light'
        extra: 'none', 'pool', or 'garage'
    
    Returns:
        PIL Image object
    """
    
    seed = random.randint(0, 2**32 - 1)
    
    # Build prompt
    prompt = "4k, highly detailed"
    
    # Size mapping
    size_map = {'small': 'small', 'medium': 'medium', 'large': 'big'}
    prompt += f", {size_map[size]}"
    
    # Style
    if style == 'modern':
        prompt += " modern villa"
    elif style == 'wooden':
        prompt += " wooden house"
    else:
        prompt += " mediterranean villa"
    
    # Story count for large modern houses
    if style == 'modern' and size == 'large':
        prompt += ", 2 story"
    
    # Modern-specific details
    if style == 'modern':
        prompt += ", white facade, interior lighting, glass panels, clean lines, minimalist design"
    
    # Roof color
    if roof_color == 'dark':
        prompt += ", dark gray slate roof"
    elif roof_color == 'red':
        prompt += ", red clay tile roof"
    else:
        prompt += ", light beige tile roof"
    
    # Extras
    if extra == 'pool':
        prompt += ", lawn and pool in front, water feature"
    elif extra == 'garage':
        prompt += ", attached garage, paved driveway"
    else:
        prompt += ", manicured lawn, garden landscaping"
    
    # Quality enhancements
    prompt += ", trees on sides, professional architectural photography, sharp focus, crisp details, pristine, photorealistic, masterpiece"
    
    # Negative prompt to avoid common issues
    negative_prompt = (
        "blurry, soft focus, out of focus, fuzzy, hazy, foggy, "
        "low quality, bad quality, worst quality, poor quality, low resolution, "
        "distorted, deformed, disfigured, ugly, gross proportions, "
        "cartoon, anime, animated, sketch, drawing, painting, illustration, "
        "rendered, 3d render, cgi, computer graphics, unrealistic, artificial, "
        "plastic, toy, miniature, fake, "
        "people, humans, persons, man, woman, child, faces, "
        "text, watermark, signature, logo, caption, letters, words, "
        "duplicate, cloned, copy, multiple, floating, levitating, "
        "out of frame, cropped, cut off, "
        "grainy, noisy, pixelated, jpeg artifacts, compression artifacts, "
        "oversaturated, undersaturated, overexposed, underexposed, "
        "amateur, unprofessional"
    )
    
    logging.info(f"🎨 '{prompt[:100]}...'")
    logging.info(f"🎲 Seed: {seed}")
    
    try:
        cleanup_memory()
        start = time.time()
        
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=12,
                guidance_scale=3.5,
                width=864,
                height=864,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images[0]
        
        logging.info(f"⏱️  {time.time() - start:.2f}s")
        return image
        
    except Exception as e:
        logging.error(f"❌ {str(e)}")
        cleanup_memory()
        raise

# ──────────────────────────────────────────────────────
# 💰 PRICE ESTIMATION
# ──────────────────────────────────────────────────────

def estimate_price(style, size, roof_color, extra):
    """
    Estimate house price based on parameters
    
    Args:
        style: House style
        size: House size
        roof_color: Roof color
        extra: Extra features
    
    Returns:
        Estimated price in euros
    """
    base_prices = {'small': 250000, 'medium': 400000, 'large': 650000}
    style_multipliers = {'modern': 1.15, 'wooden': 1.0, 'mediterranean': 1.1}
    roof_adjustments = {'dark': 0, 'red': 5000, 'light': 3000}
    
    price = base_prices.get(size, 400000)
    price *= style_multipliers.get(style, 1.0)
    price += roof_adjustments.get(roof_color, 0)
    
    if extra == 'pool':
        price += 50000
    elif extra == 'garage':
        price += 35000
    
    # Add random variation
    price *= random.uniform(0.95, 1.05)
    return int(price)

# ──────────────────────────────────────────────────────
# 🌐 API ROUTES
# ──────────────────────────────────────────────────────

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate house image based on user parameters"""
    try:
        data = request.json
        
        style = str(data.get('style', 'modern'))
        size = str(data.get('size', 'medium'))
        roof_color = str(data.get('roof_color', 'dark'))
        extra = str(data.get('extra', 'none'))
        
        logging.info(f"📥 style={style}, size={size}, roof={roof_color}, extra={extra}")
        
        # Generate image
        image = generate_house_image(style, size, roof_color, extra)
        
        # Convert to JPEG
        img_io = BytesIO()
        image.save(img_io, 'JPEG', quality=95, optimize=True)
        img_io.seek(0)
        
        logging.info(f"✅ Bild generiert ({img_io.getbuffer().nbytes / 1024:.1f} KB)")
        
        return send_file(
            img_io,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name='house.jpg'
        )
        
    except Exception as e:
        logging.error(f"❌ Fehler bei Generierung: {str(e)}", exc_info=True)
        cleanup_memory()
        return jsonify({'error': str(e)}), 500

@app.route('/estimate-price', methods=['POST'])
def estimate_price_route():
    """Estimate house price based on parameters"""
    try:
        data = request.json
        price = estimate_price(
            data.get('style', 'modern'),
            data.get('size', 'medium'),
            data.get('roof_color', 'dark'),
            data.get('extra', 'none')
        )
        return jsonify({
            'price': price,
            'formatted': f"€{price:,}".replace(',', '.')
        })
    except Exception as e:
        logging.error(f"❌ Fehler bei Preisschätzung: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        mem_alloc = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        return jsonify({
            'status': 'ok',
            'model': 'Dreamshaper XL Turbo',
            'device': torch.cuda.get_device_name(0),
            'memory_allocated_gb': round(mem_alloc, 2),
            'memory_reserved_gb': round(mem_reserved, 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

# ──────────────────────────────────────────────────────
# 🚀 RUN
# ──────────────────────────────────────────────────────

if __name__ == '__main__':
    # Enable performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    logging.info(f"🚀 Server auf http://0.0.0.0:5000")
    logging.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)