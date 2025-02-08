from enum import Enum


class Color(Enum):
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PINK = '\033[95m'
    CYAN = '\033[96m'
    GREY = '\033[97m'
    RESET = '\x1b[0m'


def colorize(text: str, color_code: Color) -> str:
    return f'{color_code.value}{text}{Color.RESET.value}'


def colorize_bool(is_green: bool, text: str = None) -> str:
    if not text:
        text = str(is_green)
    return colorize(text, Color.GREEN) if is_green else colorize(text, Color.RED)


def colorize_float(text: float, color_code: Color = Color.GREY) -> str:
    text = f'{text:.3f}'
    return colorize(text, color_code)


def colorize_str(text: str, color_code: Color = Color.GREY) -> str:
    return colorize(text, color_code)
