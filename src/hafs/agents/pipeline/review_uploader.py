import warnings
from agents.pipeline.review_uploader import ReviewUploader

warnings.warn(
    "agents.pipeline.review_uploader is deprecated. Import from 'agents.pipeline.review_uploader' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export
ReviewUploader = ReviewUploader
