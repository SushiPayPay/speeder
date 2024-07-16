from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

ds = load_dataset('timm/imagenet-1k-wds', cache_dir='/data/clean/imagenet-raw')
