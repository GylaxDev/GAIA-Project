"""HUD, menus, maps, settings pages and shared interface widgets."""

from .radial_wheel import (
    RadialWheel,
    RadialAction,
    RadialWheelState,
    RadialWheelManager,
    create_hacking_wheel,
    create_vehicle_wheel,
    create_combat_wheel,
)

# Watch Dogs themed UI system
from .watchdogs_theme import (
    WD_CYAN, WD_CYAN_SOFT, WD_CYAN_FAINT, WD_CYAN_DIM, WD_CYAN_BRIGHT,
    WD_ORANGE, WD_GREEN, WD_PURPLE,
    WD_BG_DEEP, WD_BG_DARK, WD_BG_MEDIUM, WD_BG_LIGHT,
    WD_PANEL_GRADIENT, WD_CARD_GRADIENT, WD_BUTTON_GRADIENT,
    WD_BUTTON_HOVER_GRADIENT, WD_BUTTON_PRESS_GRADIENT,
    WD_BORDER_PRIMARY, WD_BORDER_SECONDARY, WD_BORDER_SUBTLE, WD_BORDER_FOCUS,
    WD_TEXT_PRIMARY, WD_TEXT_SECONDARY, WD_TEXT_MUTED, WD_TEXT_DISABLED,
    WD_OK, WD_WARN, WD_ERROR,
    WD_R_SMALL, WD_R_MEDIUM, WD_R_LARGE, WD_R_PILL, WD_R_CARD,
    WD_T_MICRO, WD_T_CAPTION, WD_T_SMALL, WD_T_BODY, WD_T_BASE, WD_T_LABEL, WD_T_HEAD, WD_DISPLAY,
    lerp, lerp_color, ease_out, stagger, glow_color,
    COMPAT,
)
from .watchdogs_icons import draw_wd, WD_ICONS
from .watchdogs_ui import (
    wd_brackets, wd_scanlines, wd_gradient_panel, wd_gradient_button,
    wd_toggle, wd_slider, wd_segmented, wd_card, wd_heading,
    wd_pill_chip, wd_scrim, wd_prompt_bar, wd_progress_bar,
)
from .watchdogs_settings import WatchDogsSettingsPage
from .watchdogs_hud import (
    WatchDogsMinimap, WatchDogsSpeedCard, WatchDogsStatusCard,
    WatchDogsCompass, WatchDogsTimeWeatherCard, WatchDogsNotification,
)

__all__ = [
    "RadialWheel",
    "RadialAction",
    "RadialWheelState",
    "RadialWheelManager",
    "create_hacking_wheel",
    "create_vehicle_wheel",
    "create_combat_wheel",
    # Watch Dogs theme
    "WD_CYAN", "WD_CYAN_SOFT", "WD_CYAN_FAINT", "WD_CYAN_DIM", "WD_CYAN_BRIGHT",
    "WD_ORANGE", "WD_GREEN", "WD_PURPLE",
    "WD_BG_DEEP", "WD_BG_DARK", "WD_BG_MEDIUM", "WD_BG_LIGHT",
    "WD_PANEL_GRADIENT", "WD_CARD_GRADIENT", "WD_BUTTON_GRADIENT",
    "WD_BUTTON_HOVER_GRADIENT", "WD_BUTTON_PRESS_GRADIENT",
    "WD_BORDER_PRIMARY", "WD_BORDER_SECONDARY", "WD_BORDER_SUBTLE", "WD_BORDER_FOCUS",
    "WD_TEXT_PRIMARY", "WD_TEXT_SECONDARY", "WD_TEXT_MUTED", "WD_TEXT_DISABLED",
    "WD_OK", "WD_WARN", "WD_ERROR",
    "WD_R_SMALL", "WD_R_MEDIUM", "WD_R_LARGE", "WD_R_PILL", "WD_R_CARD",
    "WD_T_MICRO", "WD_T_CAPTION", "WD_T_SMALL", "WD_T_BODY", "WD_T_BASE", "WD_T_LABEL", "WD_T_HEAD", "WD_DISPLAY",
    "lerp", "lerp_color", "ease_out", "stagger", "glow_color",
    "COMPAT",
    # Watch Dogs icons
    "draw_wd", "WD_ICONS",
    # Watch Dogs UI components
    "wd_brackets", "wd_scanlines", "wd_gradient_panel", "wd_gradient_button",
    "wd_toggle", "wd_slider", "wd_segmented", "wd_card", "wd_heading",
    "wd_pill_chip", "wd_scrim", "wd_prompt_bar", "wd_progress_bar",
    # Watch Dogs pages
    "WatchDogsSettingsPage",
    # Watch Dogs HUD
    "WatchDogsMinimap", "WatchDogsSpeedCard", "WatchDogsStatusCard",
    "WatchDogsCompass", "WatchDogsTimeWeatherCard", "WatchDogsNotification",
]
