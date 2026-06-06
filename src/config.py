from pathlib import Path
import os


def authorization_header(token: str) -> str:
    """Return an Authorization header value for an API token."""
    if not token:
        return ""
    return token if token.lower().startswith("bearer ") else f"Bearer {token}"


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
PROJECT_ROOT = str(PROJECT_ROOT)
LOG_DIR = str(LOG_DIR)

# Device for optional SAM3 and GroundingDINO tools.
DEVICE = os.getenv("IEA_DEVICE", "cuda")
# Maximum number of masks and boxes selected for one image.
Top_K = 30

# LLM configuration
LLM_SERVER = os.getenv("IEA_LLM_SERVER", "vllm")
LLM_DEFAULT_MODEL = os.getenv("IEA_LLM_MODEL", "OpenDFM/Qwen2.5-VL-7B-IEA")
VLM_DEFAULT_MODEL = os.getenv("IEA_VLM_MODEL", LLM_DEFAULT_MODEL)
LLM_BASE_URL = os.getenv(
    "IEA_LLM_BASE_URL",
    "http://localhost:18181/v1/chat/completions",
)
LLM_API_KEY = authorization_header(os.getenv("IEA_LLM_API_KEY", "iea-local"))
LLM_API_TIMEOUT = int(os.getenv("IEA_LLM_TIMEOUT", "60"))
LLM_MAX_RETRY = int(os.getenv("IEA_LLM_MAX_RETRY", "3"))
VLM_MAX_IMAGE_NUMBER = int(os.getenv("IEA_VLM_MAX_IMAGE_NUMBER", "1"))
LLM_MAX_RESPONSE_TOKENS = int(os.getenv("IEA_LLM_MAX_RESPONSE_TOKENS", "1024"))

RM_DEFAULT_MODEL = os.getenv("IEA_RM_MODEL", "OpenDFM/Qwen3-0.6B-IEA-RM")
RM_BASE_URL = os.getenv(
    "IEA_RM_BASE_URL",
    "http://localhost:18182/v1/chat/completions",
)
RM_API_KEY = authorization_header(os.getenv("IEA_RM_API_KEY", "iea-local"))
RM_MAX_RESPONSE_TOKENS = int(os.getenv("IEA_RM_MAX_RESPONSE_TOKENS", "10"))
RM_MAX_RETRY = int(os.getenv("IEA_RM_MAX_RETRY", "3"))
RM_VOTE_TIMES = int(os.getenv("IEA_RM_VOTE_TIMES", "5"))



# Image processing configuration
IMAGE_RESIZE = -1 # set to -1 to disable image resize
IMAGE_RESIZE_BASE = "MAX"  # Larger size to resize the image to, options: "MAX", "MIN", "SQUARE"
NUMERICAL_OPERATORS = {
    'exposure': (-100, 100),
    'brightness': (-100, 100),
    'contrast': (-100, 100),
    'natural_contrast': (-100, 100),
    'highlights': (-100, 100),
    'shadows': (-100, 100),
    'whites': (-100, 100),
    'blacks': (-100, 100),
    'saturation': (-100, 100),
    'vibrance': (-100, 100),
    'temperature': (-100, 100),
    'tint': (-100, 100),
    'sharpness': (-100, 100),
    'vignette': (-100, 100),
    'fade': (-100, 100),
    'grain': (-100, 100),
}
NON_NUMERICAL_OPERATORS = [
    'inpaint_obj', 'deform_obj', 'color_bg', 'crop', 'black&white',
    'flip', 'edge', 'rotate', 'radial_blur', 'facet_blur', 'gaussian_blur'
]
REPLACE_TABLE={
    "sharpen": "sharpness",
    "hue": "temperature",
    "lightness": "highlights",
}

# Simulated Annealing configuration
SA_MAX_ITER = 1000  # 最大迭代次数
SA_PATIENCE = 10  # 无优化次数
SA_INITIAL_TEMP = 10.0  # 初始温度
SA_FINAL_TEMP = 0.1  # 最终温度
SA_PERTURBATION_BASE = 50  # 扰动的基础幅度
SA_PARAM_CHANGE_PROB = 0.5  # 参数改变概率

# Pipeline configuration
REFLECT_MAX_ITER = 0  # 最大反思迭代次数
