import os
from modules import paths


def preload(parser):
    parser.add_argument("--beautifulprompt-dir", type=str, help="Path to directory with BeautilfulPrompt models.", default=None)
