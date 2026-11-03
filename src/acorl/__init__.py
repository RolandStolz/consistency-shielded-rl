import os
import warnings

warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
