import streamlit as st
import torch
from PIL import Image
import io
import os

# Disable problematic PyTorch backends to prevent dynamo errors
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Enable synchronous CUDA for better error reporting

# Set torch backends to avoid compilation issues
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Tentative d'import d'Unsloth avec fallback
try:
    from unsloth import FastVisionModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    st.error("⚠️ Unsloth non disponible. Tentative avec transformers standard...")
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from transformers import TextStreamer
except ImportError:
    st.warning("TextStreamer non disponible, utilisation du mode standard")

# Configuration de la page
st.set_page_config(
    page_title="Car Damage Assessment",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    
    .analysis-result {
        background: #f9fafb;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 0.9rem;
    }
    
    .severity-minor {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 0.75rem;
        border-radius: 4px;
        color: #065f46;
    }
    
    .severity-moderate {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 0.75rem;
        border-radius: 4px;
        color: #92400e;
    }
    
    .severity-major {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 0.75rem;
        border-radius: 4px;
        color: #991b1b;
    }
    
    .upload-area {
        border: 2px dashed #d1d5db;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #fafafa;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #3b82f6;
        background: #f0f9ff;
    }
    
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Configuration du device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Fonction de nettoyage mémoire
def clear_gpu_memory():
    """Nettoie la mémoire GPU pour éviter les problèmes de mémoire"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Afficher les informations GPU dans la sidebar
def display_gpu_info():
    """Affiche les informations GPU dans la sidebar"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_cached = torch.cuda.memory_reserved(0) / 1024**3
        
        return f"""
        **GPU Information**
        - Device: {gpu_name}
        - Total Memory: {gpu_memory:.1f} GB
        - Allocated: {gpu_allocated:.1f} GB
        - Cached: {gpu_cached:.1f} GB
        - Available: {gpu_memory - gpu_cached:.1f} GB
        """
    return "**GPU Information**\n- No CUDA device available"

# Titre principal
st.markdown('<h1 class="main-header">Car Damage Assessment</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Professional AI-powered vehicle damage analysis system</p>', unsafe_allow_html=True)

# Sidebar pour les informations
with st.sidebar:
    st.markdown("### System Information")
    
    # Statut du modèle
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("**Model Status**")
    if UNSLOTH_AVAILABLE:
        st.success("Unsloth Framework: Active")
    else:
        st.info("Standard Transformers: Active")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Fonctionnalités
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("**Key Features**")
    st.markdown("""
    - Automated damage detection
    - Severity classification
    - Multi-language support
    - Professional-grade analysis
    - Real-time processing
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Spécifications techniques
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("**Technical Specifications**")
    st.markdown("""
    - **Base Model**: Llama-3.2-11B-Vision
    - **Specialization**: Automotive damage assessment
    - **Precision**: Professional level
    - **Languages**: French, English
    - **Input**: Images (PNG, JPG, JPEG)
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Informations GPU
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown(display_gpu_info())
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Bouton de nettoyage mémoire
    if st.button("Clear GPU Memory", help="Free up GPU memory"):
        clear_gpu_memory()
        st.success("GPU memory cleared!")

# Cache pour le modèle
@st.cache_resource
def load_model():
    """Charge le modèle de détection des dommages"""
    with st.spinner("Loading AI model..."):
        try:
            if UNSLOTH_AVAILABLE:
                # Utiliser Unsloth avec optimisations mémoire
                model, tokenizer = FastVisionModel.from_pretrained(
                    model_name="Kakyoin03/car-damage-assessment-llama-vision",
                    load_in_4bit=True,
                    device_map="auto",
                    max_memory={0: "40GB"},  # Limite explicite pour L40S
                )
                FastVisionModel.for_inference(model)
                st.success("Unsloth model loaded successfully!")
            else:
                # Fallback vers transformers standard avec optimisations
                st.info("Loading with standard transformers...")
                model = AutoModelForCausalLM.from_pretrained(
                    "Kakyoin03/car-damage-assessment-llama-vision",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    max_memory={0: "40GB", "cpu": "20GB"}
                )
                tokenizer = AutoProcessor.from_pretrained(
                    "Kakyoin03/car-damage-assessment-llama-vision"
                )
                st.success("Standard model loaded successfully!")
            
            return model, tokenizer
            
        except Exception as e:
            st.error(f"Model loading error: {e}")
            st.info("Attempting lighter fallback model...")
            try:
                # Utiliser le modèle original plus léger
                st.warning("Loading lighter model for better memory usage...")
                if UNSLOTH_AVAILABLE:
                    model, tokenizer = FastVisionModel.from_pretrained(
                        model_name="KHAOULA-KH/LOra_modele",
                        load_in_4bit=True,
                        device_map="auto",
                        max_memory={0: "35GB"},
                    )
                    FastVisionModel.for_inference(model)
                else:
                    # Modèle de base le plus léger possible
                    st.warning("Using minimal base model configuration")
                    model = AutoModelForCausalLM.from_pretrained(
                        "meta-llama/Llama-3.2-11B-Vision-Instruct",
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        max_memory={0: "30GB", "cpu": "15GB"},
                        load_in_8bit=True  # Utiliser 8-bit comme fallback
                    )
                    tokenizer = AutoProcessor.from_pretrained(
                        "meta-llama/Llama-3.2-11B-Vision-Instruct"
                    )
                
                return model, tokenizer
            except Exception as e2:
                st.error(f"Critical error with fallback: {e2}")
                st.error("Unable to load any model. Please check GPU memory or try restarting.")
                st.stop()

# Instruction pour le modèle
instruction = """Vous êtes un expert en évaluation de dommages automobiles.
Décrivez uniquement les pièces visibles et endommagées sur l'image.
- Ne mentionnez rien d'invisible ou non endommagé.
- Soyez concis et précis.
- N'inventez rien.
Format :
Dommages détectés sur : [liste des pièces visibles et endommagées].
Severity : [mineur / modéré / majeur]."""

def analyze_car_damage(image, model, tokenizer):
    """Analyse les dommages sur l'image de voiture"""
    try:
        # Nettoyer la mémoire GPU avant l'analyse
        clear_gpu_memory()
        
        # Convertir l'image selon le type d'entrée
        if isinstance(image, str):
            # Chemin de fichier
            new_image = Image.open(image).convert("RGB")
        elif hasattr(image, 'read'):
            # UploadedFile de Streamlit ou objet similaire
            new_image = Image.open(image).convert("RGB")
        else:
            # Objet PIL Image déjà chargé
            new_image = image.convert("RGB")

        # Redimensionner l'image si elle est trop grande (économie mémoire)
        max_size = (1024, 1024)
        if new_image.size[0] > max_size[0] or new_image.size[1] > max_size[1]:
            new_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            st.info(f"Image resized to {new_image.size} for better processing")

        # Préparer les messages
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": new_image},
                {"type": "text", "text": instruction}
            ]}
        ]

        # Tokenisation avec gestion d'erreur améliorée
        try:
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer(
                new_image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(device)
        except Exception as tokenize_error:
            st.warning(f"Primary tokenization failed: {tokenize_error}")
            # Fallback: utiliser une approche plus simple
            try:
                st.info("Using fallback tokenization method...")
                inputs = tokenizer(
                    images=new_image,
                    text=instruction,
                    return_tensors="pt",
                    padding=True
                ).to(device)
            except Exception as fallback_error:
                st.error(f"All tokenization methods failed: {fallback_error}")
                return "Tokenization error: Unable to process image and text"

        # Vérifier la validité des inputs
        if not inputs or not hasattr(inputs, 'input_ids'):
            return "Error: Invalid tokenization result"
            
        # Vérifier les token IDs pour éviter les erreurs CUDA
        if hasattr(inputs, 'input_ids'):
            max_token_id = inputs.input_ids.max().item()
            vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50000
            if max_token_id >= vocab_size:
                st.warning(f"Token ID {max_token_id} exceeds vocab size {vocab_size}, clipping...")
                inputs.input_ids = torch.clamp(inputs.input_ids, 0, vocab_size - 1)

        # Génération avec gestion mémoire optimisée
        with torch.no_grad():
            try:
                # Disable torch.compile for generation to avoid dynamo errors
                try:
                    torch._dynamo.config.disable = True
                except:
                    pass  # In case torch._dynamo is not available
                
                # Prepare generation parameters safely
                generation_kwargs = {
                    "max_new_tokens": 64,  # Reduced for stability
                    "use_cache": False,
                    "do_sample": False,  # Use greedy decoding for stability
                }
                
                # Add pad_token_id only if available and valid
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
                elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
                
                # Add eos_token_id only if available
                if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
                
                st.info("Generating response with stable settings...")
                output = model.generate(**inputs, **generation_kwargs)
                
            except Exception as gen_error:
                st.warning(f"Stable generation failed: {gen_error}")
                st.info("Trying minimal generation settings...")
                try:
                    # Ultra-minimal fallback
                    output = model.generate(
                        inputs.input_ids if hasattr(inputs, 'input_ids') else inputs['input_ids'],
                        max_length=inputs.input_ids.size(1) + 32 if hasattr(inputs, 'input_ids') else 64,
                        do_sample=False,
                        use_cache=False
                    )
                except Exception as minimal_error:
                    st.error(f"All generation methods failed: {minimal_error}")
                    return f"Generation error: {minimal_error}"

        # Décodage
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extraire seulement la réponse (après le prompt)
        if "assistant" in decoded:
            response = decoded.split("assistant")[-1].strip()
        else:
            response = decoded.strip()
        
        # Nettoyer la mémoire après l'analyse
        clear_gpu_memory()
            
        return response
        
    except Exception as e:
        clear_gpu_memory()  # Nettoyer en cas d'erreur aussi
        return f"Analysis error: {str(e)}"

# Interface principale
st.markdown("---")

# Colonnes pour l'interface
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Image Upload")
    
    # Upload de fichier avec style moderne
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a vehicle image",
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG",
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Ou utiliser la caméra
    st.markdown("### Camera Capture")
    camera_image = st.camera_input("Take a photo", label_visibility="collapsed")
    
    # Sélectionner l'image à utiliser
    selected_image = uploaded_file if uploaded_file else camera_image
    
    if selected_image:
        # Afficher l'image
        image = Image.open(selected_image)
        st.image(image, caption="Selected Image", use_container_width=True)
        
        # Informations sur l'image dans une métrique card
        st.markdown(f'''
        <div class="metric-card">
            <strong>Image Information</strong><br>
            Dimensions: {image.size[0]} × {image.size[1]} pixels<br>
            Format: {image.format}<br>
            Mode: {image.mode}
        </div>
        ''', unsafe_allow_html=True)

with col2:
    st.markdown("### Damage Analysis")
    
    if selected_image:
        # Charger le modèle
        model, tokenizer = load_model()
        
        # Bouton d'analyse avec style moderne
        if st.button("Analyze Damage", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                # Analyser l'image
                result = analyze_car_damage(selected_image, model, tokenizer)
                
                # Afficher le résultat
                st.success("Analysis completed!")
                
                # Zone de résultat stylisée
                st.markdown("### Analysis Report")
                st.markdown(f'<div class="analysis-result">{result}</div>', unsafe_allow_html=True)
                
                # Extraire la sévérité avec style moderne
                if "majeur" in result.lower() or "major" in result.lower():
                    st.markdown('<div class="severity-major"><strong>Severity:</strong> Major damage detected</div>', unsafe_allow_html=True)
                elif "modéré" in result.lower() or "moderate" in result.lower():
                    st.markdown('<div class="severity-moderate"><strong>Severity:</strong> Moderate damage detected</div>', unsafe_allow_html=True)
                elif "mineur" in result.lower() or "minor" in result.lower():
                    st.markdown('<div class="severity-minor"><strong>Severity:</strong> Minor damage detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="metric-card">
            <strong>Instructions:</strong><br>
            Please upload an image or take a photo to begin analysis
        </div>
        ''', unsafe_allow_html=True)

# Section d'exemples
st.markdown("---")
st.markdown("### Damage Categories")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    st.markdown('''
    <div class="feature-card">
        <h4 style="color: #10b981; margin-top: 0;">Minor Damage</h4>
        <ul style="margin-bottom: 0;">
            <li>Surface scratches</li>
            <li>Paint chips</li>
            <li>Small dents</li>
            <li>Minor scuffs</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

with example_col2:
    st.markdown('''
    <div class="feature-card">
        <h4 style="color: #f59e0b; margin-top: 0;">Moderate Damage</h4>
        <ul style="margin-bottom: 0;">
            <li>Medium dents</li>
            <li>Deep scratches</li>
            <li>Bumper cracks</li>
            <li>Panel deformation</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

with example_col3:
    st.markdown('''
    <div class="feature-card">
        <h4 style="color: #ef4444; margin-top: 0;">Major Damage</h4>
        <ul style="margin-bottom: 0;">
            <li>Structural deformation</li>
            <li>Broken glass</li>
            <li>Multiple damage areas</li>
            <li>Safety concerns</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>
        Powered by AI | Built with Streamlit
    </div>
    <div style='font-size: 0.9rem;'>
        Model: <a href='https://huggingface.co/Kakyoin03/car-damage-assessment-llama-vision' 
                 target='_blank' style='color: #3b82f6; text-decoration: none;'>
                Kakyoin03/car-damage-assessment-llama-vision
               </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Instructions d'utilisation
with st.expander("User Guide & Tips"):
    st.markdown("""
    ### How to use this application:
    
    1. **Image Upload**: Upload a vehicle image or take a photo using the camera
    2. **Analysis**: Click "Analyze Damage" to process the image
    3. **Results**: Review the detailed damage assessment report
    4. **Export**: Copy the results for documentation purposes
    
    ### Tips for optimal results:
    
    - Use clear, well-lit images
    - Focus on damaged areas
    - Avoid excessive reflections or glare
    - Multiple angles provide comprehensive analysis
    - Ensure the vehicle damage is clearly visible
    
    ### Technical Information:
    
    - **Processing Time**: Typically 10-30 seconds per image
    - **Accuracy**: Professional-grade assessment
    - **Languages**: French and English output
    - **File Formats**: PNG, JPG, JPEG supported
    - **Max File Size**: 200MB per image
    """)
